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


/* ------------------------------------------------------------------
   CPU Implementations
   ----------------------------------------------------------------*/

int dft_cpu(const std::complex<float>* in, std::complex<float>* out, std::size_t N)
{
    for (std::size_t k = 0; k < N; ++k)
    {
        std::complex<float> sum = 0.0f;
        for (std::size_t n = 0; n < N; ++n) sum += std::exp(std::complex<float>(0.0f, -2 * M_PI * (static_cast<float>(k) * n) / N));
        out[k] = sum;
    }
    return 0;
}

int fft_cpu(const std::complex<float>* input, std::complex<float>* output, std::size_t N)
{
    if (N == 0 || (N & (N - 1))) return -1;

    const int lgN = static_cast<int>(std::log2(N));

    for (std::size_t i = 0; i < N; ++i) output[bit_reverse(i, lgN)] = input[i];

    for (int m = 2; m <= N; m *= 2)
    {
        std::size_t mh = m >> 1;
        std::complex<float> w_m = std::exp(std::complex<float>(0.0f, -M_PI) / static_cast<float>(mh));

        for (std::size_t k = 0; k < N; k += m)
        {
            std::complex<float> w = 1.0f;
            for (std::size_t j = 0; j < mh; ++j)
            {
                auto u = output[k + j];
                auto t = w * output[k + j + mh];
                output[k + j] = u + t;
                output[k + j + mh] = u - t;
                w *= w_m;
            }
        }
    }
    return 0;
}

int fft2d_cpu(const std::complex<float>* input, std::complex<float>* output, std::size_t   rows, std::size_t cols)
{
    if (rows == 0 || cols == 0 || (rows & (rows - 1)) || (cols & (cols - 1))) return -1;

    std::vector<std::complex<float>> tmp(rows * cols);

    for (std::size_t r = 0; r < rows; ++r)
        fft_cpu(input + r * cols, tmp.data() + r * cols, cols);

    std::vector<std::complex<float>> col_in(rows), col_out(rows);
    for (std::size_t c = 0; c < cols; ++c)
    {
        for (std::size_t r = 0; r < rows; ++r)  col_in[r] = tmp[r * cols + c];
        fft_cpu(col_in.data(), col_out.data(), rows);
        for (std::size_t r = 0; r < rows; ++r)  output[r * cols + c] = col_out[r];
    }
    return 0;
}


/* ------------------------------------------------------------------
   GPU Implementations
   ----------------------------------------------------------------*/

__global__ void fft_stage_kernel(cuFloatComplex* data, std::size_t N, int stage)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t mh = 1u << (stage - 1);
    std::size_t m = mh << 1;

    std::size_t k = (tid / mh) * m;
    std::size_t j = tid % mh;

    float angle = -M_PI * static_cast<float>(j) / mh;
    float sr, cr;
    sincosf(angle, &sr, &cr);
    cuFloatComplex w = make_cuFloatComplex(cr, sr);

    cuFloatComplex u = data[k + j];
    cuFloatComplex t = cuCmulf(w, data[k + j + mh]);
    data[k + j] = cuCaddf(u, t);
    data[k + j + mh] = cuCsubf(u, t);
}

__global__ void row_fft_stage_kernel(cuFloatComplex* data, std::size_t rows, std::size_t cols, int stage)
{
    std::size_t halfPerRow = cols >> 1;   
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = rows * halfPerRow;
    if (tid >= total) return;

    std::size_t row = tid / halfPerRow;       
    std::size_t offset = tid % halfPerRow;     

    std::size_t mh = 1u << (stage - 1);
    std::size_t m = mh << 1;

    std::size_t k = (offset / mh) * m;
    std::size_t j = offset % mh;

    float angle = -M_PI * static_cast<float>(j) / static_cast<float>(mh);
    float sr, cr;  sincosf(angle, &sr, &cr);
    cuFloatComplex w = make_cuFloatComplex(cr, sr);

    std::size_t base = row * cols;
    cuFloatComplex u = data[base + k + j];
    cuFloatComplex t = cuCmulf(w, data[base + k + j + mh]);
    data[base + k + j] = cuCaddf(u, t);
    data[base + k + j + mh] = cuCsubf(u, t);
}

__global__ void col_fft_stage_kernel(cuFloatComplex* data, std::size_t rows, std::size_t cols, int stage)
{
    std::size_t halfPerCol = rows >> 1;    
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = cols * halfPerCol;
    if (tid >= total) return;

    std::size_t col = tid / halfPerCol;
    std::size_t offset = tid % halfPerCol;

    std::size_t mh = 1u << (stage - 1);
    std::size_t m = mh << 1;

    std::size_t k = (offset / mh) * m;
    std::size_t j = offset % mh;

    float angle = -M_PI * static_cast<float>(j) / static_cast<float>(mh);
    float sr, cr;  sincosf(angle, &sr, &cr);
    cuFloatComplex w = make_cuFloatComplex(cr, sr);

    std::size_t idxA = (k + j) * cols + col;
    std::size_t idxB = (k + j + mh) * cols + col;

    cuFloatComplex u = data[idxA];
    cuFloatComplex t = cuCmulf(w, data[idxB]);
    data[idxA] = cuCaddf(u, t);
    data[idxB] = cuCsubf(u, t);
}

__global__ void bit_reverse_kernel(const cuFloatComplex* in, cuFloatComplex* out, int lgN)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (1u << lgN)) {
        unsigned rev = bit_reverse(i, lgN);
        out[i] = in[rev];
    }
}

__global__ void row_bit_reverse_kernel(const cuFloatComplex* in, cuFloatComplex* out, std::size_t rows, std::size_t cols, int lgCols)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = rows * cols;
    if (idx >= total) return;

    std::size_t row = idx / cols;
    std::size_t col = idx % cols;
    unsigned    rev = bit_reverse(col, lgCols);
    out[row * cols + rev] = in[idx];
}

__global__ void col_bit_reverse_kernel(const cuFloatComplex* in, cuFloatComplex* out, std::size_t rows, std::size_t cols, int lgRows)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = rows * cols;
    if (idx >= total) return;

    std::size_t row = idx / cols;
    std::size_t col = idx % cols;
    unsigned    rev = bit_reverse(row, lgRows);
    out[rev * cols + col] = in[idx];
}
/* ------------------------------------------------------------------
   Host wrapper for GPU FFT
   ----------------------------------------------------------------*/

int fft_gpu(const std::complex<float>* host_in, std::complex<float>* host_out, std::size_t N)
{
    if (N == 0 || (N & (N - 1))) return -1;

    const int    lgN = static_cast<int>(std::log2(N));
    const size_t bytes = N * sizeof(cuFloatComplex);

    cuFloatComplex* d_in, * d_data;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_data, bytes);

    cudaMemcpy(d_in, host_in, bytes, cudaMemcpyHostToDevice);
    dim3 threads(256), blocks((N + 255) / 256);
    bit_reverse_kernel << <blocks, threads >> > (d_in, d_data, lgN);
    
    size_t half = N >> 1;
    dim3 blocks2((half + 255) / 256);
    for (int s = 1; s <= lgN; ++s)
    {
        fft_stage_kernel << <blocks2, threads >> > (d_data, N, s);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(host_out, d_data, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_data);
    return 0;
}

int fft2d_gpu(const std::complex<float>* host_in, std::complex<float>* host_out, std::size_t rows, std::size_t cols)
{
    if (rows == 0 || cols == 0 || (rows & (rows - 1)) || (cols & (cols - 1))) return -1;

    int lgRows = static_cast<int>(std::log2(rows));
    int lgCols = static_cast<int>(std::log2(cols));

    std::size_t total = rows * cols;
    std::size_t bytes = total * sizeof(cuFloatComplex);

    cuFloatComplex* d_in, * d_data1, * d_data2;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_data1, bytes);
    cudaMalloc(&d_data2, bytes);

    cudaMemcpy(d_in, host_in, bytes, cudaMemcpyHostToDevice);

    dim3 threads(256);
    dim3 blocks((total + 255) / 256);

    /* ---- Stage 1 :  row‑wise FFT --------------------------- */
    row_bit_reverse_kernel << <blocks, threads >> > (d_in, d_data1, rows, cols, lgCols);
    cudaDeviceSynchronize();

    std::size_t halfRowTotal = rows * (cols >> 1);
    dim3 blocks_row((halfRowTotal + 255) / 256);
    for (int s = 1; s <= lgCols; ++s)
    {
        row_fft_stage_kernel << <blocks_row, threads >> > (d_data1, rows, cols, s);
        cudaDeviceSynchronize();
    }

    /* ---- Stage 2 :  column‑wise FFT ------------------------ */
    col_bit_reverse_kernel << <blocks, threads >> > (d_data1, d_data2, rows, cols, lgRows);
    cudaDeviceSynchronize();

    std::size_t halfColTotal = (rows >> 1) * cols;
    dim3 blocks_col((halfColTotal + 255) / 256);
    for (int s = 1; s <= lgRows; ++s)
    {
        col_fft_stage_kernel << <blocks_col, threads >> > (d_data2, rows, cols, s);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(host_out, d_data2, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);    cudaFree(d_data1);    cudaFree(d_data2);
    return 0;
}





