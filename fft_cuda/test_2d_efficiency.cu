#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <complex>

#include "fft.cu"       

int main()
{
    using clock = std::chrono::high_resolution_clock;
    using dur_ms = std::chrono::duration<double, std::milli>;

    std::cout << std::left
        << std::setw(10) << "Size"
        << std::setw(15) << "CPU(ms)"
        << std::setw(15) << "GPU(ms)" << '\n';

    for (int exp = 1; exp <= 12; ++exp)    
    {
        std::size_t N = 1u << exp;        
        std::size_t total = N * N;

        std::vector<std::complex<float>> x(total), Xcpu(total), Xgpu(total);

        for (std::size_t r = 0; r < N; ++r)
            for (std::size_t c = 0; c < N; ++c)
                x[r * N + c] = { static_cast<float>(r + 1), static_cast<float>(c + 1) };

        /* ------------------- CPU ----------------------------- */
        auto t0_cpu = clock::now();
        fft2d_cpu(x.data(), Xcpu.data(), N, N);
        auto t1_cpu = clock::now();
        double cpu_ms = dur_ms(t1_cpu - t0_cpu).count();

        /* ------------------- GPU ----------------------------- */
        auto t0_gpu = clock::now();
        fft2d_gpu(x.data(), Xgpu.data(), N, N);
        cudaDeviceSynchronize(); 
        auto t1_gpu = clock::now();
        double gpu_ms = dur_ms(t1_gpu - t0_gpu).count();

        /* ------------------- prints --------------------------- */
        std::cout << std::setw(10) << (std::to_string(N) + "x" + std::to_string(N))
            << std::setw(15) << std::fixed << std::setprecision(3) << cpu_ms
            << std::setw(15) << std::fixed << std::setprecision(3) << gpu_ms
            << '\n';
    }

    return 0;
}
