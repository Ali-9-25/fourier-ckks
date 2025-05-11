/**********************************************************************
 *  test_1d_compare_cufft.cu
 *  -------------------------------------------------------------------
 *  Benchmarks:
 *      • fft_cpu        – iterative radix-2 FFT on host    (single-prec.)
 *      • fft_gpu        – your custom CUDA implementation (fft.cu)
 *      • cuFFT          – NVIDIA’s library implementation (cufftExecC2C)
 *
 *  Prints run times in milliseconds for input sizes N = 2^k.
 *  Timing strategy
 *      – CPU code:          std::chrono::high_resolution_clock
 *      – GPU code:          cudaEvent_t (includes H↔D copies + exec)
 *********************************************************************/

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft.cu"                    // <- your routines

/* ------------ simple CUDA / cuFFT error helpers ------------------ */
#define CHECK_CUDA(call)                                             \
    do {                                                             \
        cudaError_t _e = (call);                                     \
        if (_e != cudaSuccess) {                                     \
            std::cerr << "CUDA error " << cudaGetErrorString(_e)     \
                      << " at " << __FILE__ << ':' << __LINE__       \
                      << std::endl;                                  \
            std::exit(EXIT_FAILURE);                                 \
        }                                                            \
    } while (0)

#define CHECK_CUFFT(call)                                            \
    do {                                                             \
        cufftResult _e = (call);                                     \
        if (_e != CUFFT_SUCCESS) {                                   \
            std::cerr << "cuFFT error " << _e                        \
                      << " at " << __FILE__ << ':' << __LINE__       \
                      << std::endl;                                  \
            std::exit(EXIT_FAILURE);                                 \
        }                                                            \
    } while (0)

int main()
{
    using clock   = std::chrono::high_resolution_clock;
    using dur_ms  = std::chrono::duration<double, std::milli>;

    std::cout << std::left
              << std::setw(12) << "N"
              << std::setw(15) << "FFT_CPU(ms)"
              << std::setw(15) << "FFT_GPU(ms)"
              << std::setw(15) << "cuFFT(ms)"     << '\n';

    /* ------------------------------------------------------------------
       loop over power-of-two sizes
    ------------------------------------------------------------------ */
    for (int exp = 3; exp <= 20; ++exp)
    {
        std::size_t N = 1ULL << exp;
        std::size_t bytes = N * sizeof(cuFloatComplex);

        /* ---- host-side buffers ------------------------------------ */
        std::vector<std::complex<float>> x(N),
                                         y_cpu(N),     // fft_cpu result
                                         y_gpu(N),     // custom CUDA
                                         y_cufft(N);   // cuFFT

        /* test signal (non-trivial, deterministic) ------------------ */
        for (std::size_t i = 0; i < N; ++i)
            x[i] = { std::sin(0.017453292f * static_cast<float>(i)),
                     std::cos(0.031415927f * static_cast<float>(i)) };

        /* -------------- CPU FFT timing ----------------------------- */
        auto  t0_cpu = clock::now();
        fft_cpu(x.data(), y_cpu.data(), N);
        auto  t1_cpu = clock::now();
        double ms_cpu = dur_ms(t1_cpu - t0_cpu).count();

        /* -----------------------------------------------------------
           GPU 1: custom fft_gpu   (wrapper allocates + copies)
        ----------------------------------------------------------- */
        auto  t0_gpu = clock::now();
        fft_gpu(x.data(), y_gpu.data(), N);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto  t1_gpu = clock::now();
        double ms_gpu = dur_ms(t1_gpu - t0_gpu).count();

        /* -----------------------------------------------------------
           GPU 2: cuFFT – we time: H→D copy + exec + D→H copy
        ----------------------------------------------------------- */
        cuFloatComplex *d_data = nullptr;
        CHECK_CUDA(cudaMalloc(&d_data, bytes));

        cudaEvent_t evStart, evStop;
        CHECK_CUDA(cudaEventCreate(&evStart));
        CHECK_CUDA(cudaEventCreate(&evStop));

        /* host → device */
        CHECK_CUDA(cudaMemcpy(d_data, x.data(), bytes,
                              cudaMemcpyHostToDevice));

        /* plan – its creation time is *not* included in timing
           (cuFFT documentation recommends re-using plans)           */
        cufftHandle plan;
        CHECK_CUFFT(cufftPlan1d(&plan,
                                static_cast<int>(N),
                                CUFFT_C2C,
                                /*batch*/ 1));

        CHECK_CUDA(cudaEventRecord(evStart));
        CHECK_CUFFT(cufftExecC2C(plan,
                                 d_data, d_data,
                                 CUFFT_FORWARD));        // sign = –1
        CHECK_CUDA(cudaEventRecord(evStop));
        CHECK_CUDA(cudaEventSynchronize(evStop));

        /* device → host */
        CHECK_CUDA(cudaMemcpy(y_cufft.data(), d_data, bytes,
                              cudaMemcpyDeviceToHost));

        float ms_cufft = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_cufft, evStart, evStop));

        /* cleanup */
        cufftDestroy(plan);
        cudaFree(d_data);
        cudaEventDestroy(evStart);
        cudaEventDestroy(evStop);

        /* -------------- print line --------------------------------- */
        std::cout << std::setw(12) << N
                  << std::setw(15) << std::fixed << std::setprecision(3) << ms_cpu
                  << std::setw(15) << std::fixed << std::setprecision(3) << ms_gpu
                  << std::setw(15) << std::fixed << std::setprecision(3) << ms_cufft
                  << '\n';
    }
    return 0;
}
