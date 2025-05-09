/* ===================================================================== *
       2-D convolution:  Naïve CPU  vs.  GPU FFT (row/col powers of two)
   ---------------------------------------------------------------------
   - Generate two random real images   A  and  B
   - Reference convolution on CPU (O(M² N²))
   - FFT-based convolution on GPU:
         pad-to-power-of-two → 2-D FFT → Hadamard product
         → inverse 2-D FFT → crop
   - Check correctness (max |error|) and report timings / speed-up
 * ===================================================================== */

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "fft.cu"                 /* fft2d_gpu, ifft2d_gpu, kernels   */

/* -------- helper: next power-of-two --------------------------------- */
static std::size_t next_pow2(std::size_t v)
{
    if (v == 0) return 1;
    --v;
    v |= v >> 1;  v |= v >> 2;  v |= v >> 4;
    v |= v >> 8;  v |= v >> 16; v |= v >> 32;
    return v + 1;
}

/* -------- naïve 2-D convolution on the CPU -------------------------- */
static void conv2d_cpu(const std::vector<float>& A,
                       std::size_t rowsA, std::size_t colsA,
                       const std::vector<float>& B,
                       std::size_t rowsB, std::size_t colsB,
                       std::vector<float>&       C)          // result
{
    std::size_t rowsC = rowsA + rowsB - 1;
    std::size_t colsC = colsA + colsB - 1;
    C.assign(rowsC * colsC, 0.0f);

    for (std::size_t ra = 0; ra < rowsA; ++ra)
        for (std::size_t ca = 0; ca < colsA; ++ca)
        {
            float aVal = A[ra * colsA + ca];
            for (std::size_t rb = 0; rb < rowsB; ++rb)
                for (std::size_t cb = 0; cb < colsB; ++cb)
                {
                    std::size_t r = ra + rb;
                    std::size_t c = ca + cb;
                    C[r * colsC + c] += aVal * B[rb * colsB + cb];
                }
        }
}

/* ==================================================================== */
int main(int argc, char* argv[])
{
    /* ---- choose image sizes (can be given on the CLI) ------------- */
    std::size_t RA = (argc >= 2 ? std::stoul(argv[1]) : 256);   // rows A
    std::size_t CA = (argc >= 3 ? std::stoul(argv[2]) : 256);   // cols A
    std::size_t RB = (argc >= 4 ? std::stoul(argv[3]) : 128);   // rows B
    std::size_t CB = (argc >= 5 ? std::stoul(argv[4]) : 128);   // cols B

    std::cout << "2-D convolution:\n"
              << "    A = " << RA << " x " << CA
              << ",  B = " << RB << " x " << CB << '\n';

    /* ---- generate random real inputs ------------------------------ */
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(RA * CA), B(RB * CB);
    for (float& v : A) v = dist(rng);
    for (float& v : B) v = dist(rng);

    /* ---------------- reference convolution on CPU ----------------- */
    std::vector<float> C_cpu;
    auto t0_cpu = std::chrono::high_resolution_clock::now();
    conv2d_cpu(A, RA, CA, B, RB, CB, C_cpu);
    auto t1_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1_cpu - t0_cpu).count();

    /* ------------------- FFT-based pipeline on GPU ----------------- */
    /* pad each dimension to power-of-two ≥ rowsA+rowsB-1, colsA+colsB-1 */
    std::size_t rowsF = next_pow2(RA + RB - 1);
    std::size_t colsF = next_pow2(CA + CB - 1);
    std::size_t totalF = rowsF * colsF;

    std::vector<std::complex<float>>
        Ap(rowsF * colsF), Bp(rowsF * colsF),
        FA(totalF), FB(totalF),
        FC(totalF), Cp(totalF);            // padded / frequency / temp

    /* copy inputs into the top-left corner, imag = 0 */
    for (std::size_t r = 0; r < RA; ++r)
        for (std::size_t c = 0; c < CA; ++c)
            Ap[r * colsF + c] = { A[r * CA + c], 0.f };

    for (std::size_t r = 0; r < RB; ++r)
        for (std::size_t c = 0; c < CB; ++c)
            Bp[r * colsF + c] = { B[r * CB + c], 0.f };

    auto t0_gpu = std::chrono::high_resolution_clock::now();

    /* ---- forward 2-D FFTs ---- */
    fft2d_gpu(Ap.data(), FA.data(), rowsF, colsF);
    fft2d_gpu(Bp.data(), FB.data(), rowsF, colsF);

    /* ---- Hadamard product on GPU ---- */
    cuFloatComplex *dFA{}, *dFB{}, *dFC{};
    std::size_t bytes = totalF * sizeof(cuFloatComplex);
    cudaMalloc(&dFA, bytes);
    cudaMalloc(&dFB, bytes);
    cudaMalloc(&dFC, bytes);

    cudaMemcpy(dFA, FA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dFB, FB.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(256);
    dim3 blocks((totalF + threads.x - 1) / threads.x);
    pointwise_mul_2d_kernel<<<blocks, threads>>>(dFA, dFB, dFC, totalF);

    cudaMemcpy(FC.data(), dFC, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dFA); cudaFree(dFB); cudaFree(dFC);

    /* ---- inverse 2-D FFT ---- */
    ifft2d_gpu(FC.data(), Cp.data(), rowsF, colsF);

    cudaDeviceSynchronize();
    auto t1_gpu = std::chrono::high_resolution_clock::now();
    double gpu_ms =
        std::chrono::duration<double, std::milli>(t1_gpu - t0_gpu).count();

    /* ---- crop the valid region (rowsC × colsC) -------------------- */
    std::size_t rowsC = RA + RB - 1;
    std::size_t colsC = CA + CB - 1;

    std::vector<float> C_fft(rowsC * colsC);
    for (std::size_t r = 0; r < rowsC; ++r)
        for (std::size_t c = 0; c < colsC; ++c)
            C_fft[r * colsC + c] = Cp[r * colsF + c].real();

    /* ---------------- correctness check ---------------------------- */
    float max_err = 0.f, mse = 0.f;
    for (std::size_t i = 0, n = rowsC * colsC; i < n; ++i)
    {
        float e = std::fabs(C_cpu[i] - C_fft[i]);
        max_err = std::max(max_err, e);
        mse    += e * e;
    }
    mse /= static_cast<float>(rowsC * colsC);
    float rmse = std::sqrt(mse);

    /* ---------------- report --------------------------------------- */
    std::cout << std::fixed << std::setprecision(3)
              << "    padded FFT size   : " << rowsF << " x " << colsF << '\n'
              << "    CPU time          : " << cpu_ms << "  ms\n"
              << "    GPU FFT time      : " << gpu_ms << "  ms\n"
              << "    Speed-up (GPU/CPU): " << (cpu_ms / gpu_ms) << "x\n"
              << "    max |error|       : " << max_err << '\n'
              << "    RMSE              : " << rmse   << '\n';

    if (max_err < 1e-3f)
        std::cout << ">>> PASS\n";
    else
        std::cout << ">>> FAIL\n";

    return 0;
}
