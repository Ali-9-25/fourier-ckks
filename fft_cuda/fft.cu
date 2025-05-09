/* ================================================================== *
   Fast Fourier Transform (single-precision, radix-2, CUDA & CPU)
   — unified forward / inverse kernels (sign = ±1)
 * ================================================================== */

 #include <cuda_runtime.h>
 #include <cuComplex.h>
 
 #define _USE_MATH_DEFINES
 #include <cmath>
 #include <complex>
 #include <iostream>
 #include <vector>
 
 #ifndef M_PI
 #define M_PI 3.14159265358979323846f
 #endif
 
 
 /* ------------------------------------------------------------------ *
    Bit-reversal: used by both host & device code
  * ------------------------------------------------------------------ */
 __host__ __device__
 unsigned bit_reverse(unsigned v, int lgN)
 {
     v = ((v & 0x55555555u) << 1) | ((v & 0xAAAAAAAAu) >> 1);
     v = ((v & 0x33333333u) << 2) | ((v & 0xCCCCCCCCu) >> 2);
     v = ((v & 0x0F0F0F0Fu) << 4) | ((v & 0xF0F0F0F0u) >> 4);
     v = ((v & 0x00FF00FFu) << 8) | ((v & 0xFF00FF00u) >> 8);
     v = (v << 16) | (v >> 16);
     return v >> (32 - lgN);
 }
 
 
 /* ================================================================== *
                              CPU  ROUTINES
  * ================================================================== */
 
 /* ---------- simple, slow DFT (unchanged) -------------------------- */
 int dft_cpu(const std::complex<float>* in, std::complex<float>* out,
             std::size_t N)
 {
     for (std::size_t k = 0; k < N; ++k) {
         std::complex<float> sum = 0.0f;
         for (std::size_t n = 0; n < N; ++n)
             sum += in[n] *
                    std::exp(std::complex<float>(0.0f,
                           -2.f * M_PI * static_cast<float>(k * n) / N));
         out[k] = sum;
     }
     return 0;
 }
 
 /* ---------- generic iterative FFT (dir = −1 fwd, +1 inv) --------- */
 static int fft_cpu_generic(const std::complex<float>* input,
                            std::complex<float>*       output,
                            std::size_t                N,
                            float                      dir)
 {
     if (N == 0 || (N & (N - 1))) return -1;      /* must be power of two */
 
     int lgN = static_cast<int>(std::log2(N));
 
     for (std::size_t i = 0; i < N; ++i)
         output[bit_reverse(i, lgN)] = input[i];
 
     for (int m = 2; m <= static_cast<int>(N); m <<= 1) {
         std::size_t mh = m >> 1;
         std::complex<float> w_m =
             std::exp(std::complex<float>(0.0f, dir * M_PI)
                      / static_cast<float>(mh));
 
         for (std::size_t k = 0; k < N; k += m) {
             std::complex<float> w = 1.0f;
             for (std::size_t j = 0; j < mh; ++j) {
                 auto u = output[k + j];
                 auto t = w * output[k + j + mh];
                 output[k + j]       = u + t;
                 output[k + j + mh]  = u - t;
                 w *= w_m;
             }
         }
     }
     return 0;
 }
 
 /* ---------- public wrappers -------------------------------------- */
 int  fft_cpu (const std::complex<float>* in,
               std::complex<float>*       out,
               std::size_t                N)
 { return fft_cpu_generic(in, out, N, -1.f); }
 
 int  ifft_cpu(const std::complex<float>* in,
               std::complex<float>*       out,
               std::size_t                N)
 {
     int err = fft_cpu_generic(in, out, N, +1.f);
     if (err) return err;
     float s = 1.f / static_cast<float>(N);
     for (std::size_t i = 0; i < N; ++i) out[i] *= s;
     return 0;
 }
 
 /* ---------- 2-D helpers ------------------------------------------ */
 static int fft2d_cpu_generic(const std::complex<float>* in,
                              std::complex<float>*       out,
                              std::size_t                rows,
                              std::size_t                cols,
                              float                      dir)
 {
     if (rows == 0 || cols == 0
         || (rows & (rows - 1)) || (cols & (cols - 1)))
         return -1;
 
     std::vector<std::complex<float>> tmp(rows * cols);
 
     /* row transforms */
     for (std::size_t r = 0; r < rows; ++r)
         fft_cpu_generic(in + r * cols, tmp.data() + r * cols, cols, dir);
 
     /* column transforms */
     std::vector<std::complex<float>> col_in(rows), col_out(rows);
     for (std::size_t c = 0; c < cols; ++c) {
         for (std::size_t r = 0; r < rows; ++r)
             col_in[r] = tmp[r * cols + c];
 
         fft_cpu_generic(col_in.data(), col_out.data(), rows, dir);
 
         for (std::size_t r = 0; r < rows; ++r)
             out[r * cols + c] = col_out[r];
     }
     return 0;
 }
 
 int  fft2d_cpu (const std::complex<float>* in,
                 std::complex<float>*       out,
                 std::size_t                rows,
                 std::size_t                cols)
 { return fft2d_cpu_generic(in, out, rows, cols, -1.f); }
 
 int  ifft2d_cpu(const std::complex<float>* in,
                 std::complex<float>*       out,
                 std::size_t                rows,
                 std::size_t                cols)
 {
     int err = fft2d_cpu_generic(in, out, rows, cols, +1.f);
     if (err) return err;
     float s = 1.f / static_cast<float>(rows * cols);
     for (std::size_t i = 0, n = rows * cols; i < n; ++i) out[i] *= s;
     return 0;
 }
 
 
 /* ================================================================== *
                            GPU   K E R N E L S
  * ================================================================== */
 
 /* ---------- unified radix-2 butterfly (dir = ±1) ------------------ */
 __global__ void fft_stage_kernel(cuFloatComplex* data,
                                  std::size_t     N,
                                  int             stage,
                                  float           dir)
 {
     unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
     if (tid >= (N >> 1)) return;              /* only N/2 butterflies */
 
     std::size_t mh = 1u << (stage - 1);
     std::size_t m  = mh << 1;
 
     std::size_t k = (tid / mh) * m;
     std::size_t j = tid % mh;
 
     float angle = dir * M_PI * static_cast<float>(j) / static_cast<float>(mh);
     float sr, cr;  sincosf(angle, &sr, &cr);
     cuFloatComplex w = make_cuFloatComplex(cr, sr);
 
     cuFloatComplex u = data[k + j];
     cuFloatComplex t = cuCmulf(w, data[k + j + mh]);
     data[k + j]      = cuCaddf(u, t);
     data[k + j + mh] = cuCsubf(u, t);
 }
 
 /* ---------- row-wise butterfly ----------------------------------- */
 __global__ void row_fft_stage_kernel(cuFloatComplex* data,
                                      std::size_t     rows,
                                      std::size_t     cols,
                                      int             stage,
                                      float           dir)
 {
     std::size_t halfPerRow = cols >> 1;
     std::size_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
     std::size_t work = rows * halfPerRow;
     if (tid >= work) return;
 
     std::size_t row = tid / halfPerRow;
     std::size_t off = tid % halfPerRow;
 
     std::size_t mh = 1u << (stage - 1);
     std::size_t m  = mh << 1;
 
     std::size_t k = (off / mh) * m;
     std::size_t j =  off % mh;
 
     float angle = dir * M_PI * static_cast<float>(j) / static_cast<float>(mh);
     float sr, cr;  sincosf(angle, &sr, &cr);
     cuFloatComplex w = make_cuFloatComplex(cr, sr);
 
     std::size_t base = row * cols;
     cuFloatComplex u = data[base + k + j];
     cuFloatComplex t = cuCmulf(w, data[base + k + j + mh]);
     data[base + k + j]      = cuCaddf(u, t);
     data[base + k + j + mh] = cuCsubf(u, t);
 }
 
 /* ---------- column-wise butterfly -------------------------------- */
 __global__ void col_fft_stage_kernel(cuFloatComplex* data,
                                      std::size_t     rows,
                                      std::size_t     cols,
                                      int             stage,
                                      float           dir)
 {
     std::size_t halfPerCol = rows >> 1;
     std::size_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
     std::size_t work = cols * halfPerCol;
     if (tid >= work) return;
 
     std::size_t col = tid / halfPerCol;
     std::size_t off = tid % halfPerCol;
 
     std::size_t mh = 1u << (stage - 1);
     std::size_t m  = mh << 1;
 
     std::size_t k = (off / mh) * m;
     std::size_t j =  off % mh;
 
     float angle = dir * M_PI * static_cast<float>(j) / static_cast<float>(mh);
     float sr, cr;  sincosf(angle, &sr, &cr);
     cuFloatComplex w = make_cuFloatComplex(cr, sr);
 
     std::size_t idxA = (k + j)      * cols + col;
     std::size_t idxB = (k + j + mh) * cols + col;
 
     cuFloatComplex u = data[idxA];
     cuFloatComplex t = cuCmulf(w, data[idxB]);
     data[idxA] = cuCaddf(u, t);
     data[idxB] = cuCsubf(u, t);
 }
 
 /* ---------- bit-reverse reorder kernels (unchanged) -------------- */
 __global__ void bit_reverse_kernel(const cuFloatComplex* in,
                                    cuFloatComplex*       out,
                                    int                   lgN)
 {
     unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
     unsigned n = 1u << lgN;
     if (i >= n) return;
     out[i] = in[bit_reverse(i, lgN)];
 }
 
 __global__ void row_bit_reverse_kernel(const cuFloatComplex* in,
                                        cuFloatComplex*       out,
                                        std::size_t           rows,
                                        std::size_t           cols,
                                        int                   lgCols)
 {
     std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
     std::size_t total = rows * cols;
     if (idx >= total) return;
 
     std::size_t row = idx / cols;
     std::size_t col = idx % cols;
     unsigned rev = bit_reverse(static_cast<unsigned>(col), lgCols);
     out[row * cols + rev] = in[idx];
 }
 
 __global__ void col_bit_reverse_kernel(const cuFloatComplex* in,
                                        cuFloatComplex*       out,
                                        std::size_t           rows,
                                        std::size_t           cols,
                                        int                   lgRows)
 {
     std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
     std::size_t total = rows * cols;
     if (idx >= total) return;
 
     std::size_t row = idx / cols;
     std::size_t col = idx % cols;
     unsigned rev = bit_reverse(static_cast<unsigned>(row), lgRows);
     out[rev * cols + col] = in[idx];
 }
 
 /* ---------- scaling kernel (used only for inverse) --------------- */
 __global__ void scale_kernel(cuFloatComplex* data,
                              std::size_t     N,
                              float           s)
 {
     unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= N) return;
     data[i].x *= s;
     data[i].y *= s;
 }
 
 
 /* ================================================================== *
                      GPU HOST-SIDE  (1-D  forward / inverse)
  * ================================================================== */
 static int fft_gpu_dir(const std::complex<float>* host_in,
                        std::complex<float>*       host_out,
                        std::size_t                N,
                        float                      dir)
 {
     if (N == 0 || (N & (N - 1))) return -1;
 
     int    lgN   = static_cast<int>(std::log2(N));
     size_t bytes = N * sizeof(cuFloatComplex);
 
     cuFloatComplex *d_in{}, *d_data{};
     cudaMalloc(&d_in,   bytes);
     cudaMalloc(&d_data, bytes);
     cudaMemcpy(d_in, host_in, bytes, cudaMemcpyHostToDevice);
 
     /* --- bit-reverse copy --------------------------------------- */
     dim3 threads(256), blocks((N + 255) / 256);
     bit_reverse_kernel<<<blocks, threads>>>(d_in, d_data, lgN);
 
     /* --- iterative stages --------------------------------------- */
     size_t half = N >> 1;
     dim3 blocks2((half + 255) / 256);
     for (int s = 1; s <= lgN; ++s)
         fft_stage_kernel<<<blocks2, threads>>>(d_data, N, s, dir);
 
     /* --- scale if inverse --------------------------------------- */
     if (dir > 0.f)
         scale_kernel<<<blocks, threads>>>(d_data, N,
                                           1.f / static_cast<float>(N));
 
     /* --- copy back & cleanup ------------------------------------ */
     cudaMemcpy(host_out, d_data, bytes, cudaMemcpyDeviceToHost);
     cudaFree(d_in);   cudaFree(d_data);
     return 0;
 }
 
 int fft_gpu (const std::complex<float>* host_in,
              std::complex<float>*       host_out,
              std::size_t                N)
 { return fft_gpu_dir(host_in, host_out, N, -1.f); }
 
 int ifft_gpu(const std::complex<float>* host_in,
              std::complex<float>*       host_out,
              std::size_t                N)
 { return fft_gpu_dir(host_in, host_out, N, +1.f); }
 
 
 /* ================================================================== *
               GPU HOST-SIDE  (2-D  forward / inverse)                 *
  * ================================================================== */
 static int fft2d_gpu_dir(const std::complex<float>* host_in,
                          std::complex<float>*       host_out,
                          std::size_t                rows,
                          std::size_t                cols,
                          float                      dir)
 {
     if (rows == 0 || cols == 0
         || (rows & (rows - 1)) || (cols & (cols - 1)))
         return -1;
 
     int lgRows = static_cast<int>(std::log2(rows));
     int lgCols = static_cast<int>(std::log2(cols));
 
     std::size_t total = rows * cols;
     std::size_t bytes = total * sizeof(cuFloatComplex);
 
     cuFloatComplex *d_in{}, *d_data1{}, *d_data2{};
     cudaMalloc(&d_in,    bytes);
     cudaMalloc(&d_data1, bytes);
     cudaMalloc(&d_data2, bytes);
     cudaMemcpy(d_in, host_in, bytes, cudaMemcpyHostToDevice);
 
     dim3 threads(256);
     dim3 blocksTot((total + 255) / 256);
 
     /* --- row bit-reverse --------------------------------------- */
     row_bit_reverse_kernel<<<blocksTot, threads>>>
         (d_in, d_data1, rows, cols, lgCols);
 
     /* --- row stages -------------------------------------------- */
     std::size_t halfRowWork = rows * (cols >> 1);
     dim3 blocksRow((halfRowWork + 255) / 256);
     for (int s = 1; s <= lgCols; ++s)
         row_fft_stage_kernel<<<blocksRow, threads>>>
             (d_data1, rows, cols, s, dir);
 
     /* --- column bit-reverse ------------------------------------ */
     col_bit_reverse_kernel<<<blocksTot, threads>>>
         (d_data1, d_data2, rows, cols, lgRows);
 
     /* --- column stages ----------------------------------------- */
     std::size_t halfColWork = (rows >> 1) * cols;
     dim3 blocksCol((halfColWork + 255) / 256);
     for (int s = 1; s <= lgRows; ++s)
         col_fft_stage_kernel<<<blocksCol, threads>>>
             (d_data2, rows, cols, s, dir);
 
     /* --- scale if inverse -------------------------------------- */
     if (dir > 0.f)
         scale_kernel<<<blocksTot, threads>>>
             (d_data2, total, 1.f / static_cast<float>(total));
 
     cudaMemcpy(host_out, d_data2, bytes, cudaMemcpyDeviceToHost);
     cudaFree(d_in); cudaFree(d_data1); cudaFree(d_data2);
     return 0;
 }
 
 int fft2d_gpu (const std::complex<float>* host_in,
                std::complex<float>*       host_out,
                std::size_t                rows,
                std::size_t                cols)
 { return fft2d_gpu_dir(host_in, host_out, rows, cols, -1.f); }
 
 int ifft2d_gpu(const std::complex<float>* host_in,
                std::complex<float>*       host_out,
                std::size_t                rows,
                std::size_t                cols)
 { return fft2d_gpu_dir(host_in, host_out, rows, cols, +1.f); }
 
 /* ---------- element-wise (Hadamard) product ----------------------- */
/*  C[i] = A[i] * B[i]   for 0 ≤ i < N                                */
__global__ void pointwise_mul_kernel(const cuFloatComplex* A,
    const cuFloatComplex* B,
    cuFloatComplex*       C,
    std::size_t           N)
{
unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= N) return;
C[i] = cuCmulf(A[i], B[i]);
}

/* ---------- element–wise product for 2-D FFT data ------------------- */
/*  C[idx] = A[idx] * B[idx]  for idx = 0 … rows*cols-1                 */
__global__ void pointwise_mul_2d_kernel(const cuFloatComplex* A,
    const cuFloatComplex* B,
    cuFloatComplex*       C,
    std::size_t           total)
{
unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < total)
C[idx] = cuCmulf(A[idx], B[idx]);
}
/*  (You may also just call pointwise_mul_kernel with total=Nrows*Ncols) */
