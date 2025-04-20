#include <chrono>
#include <iomanip>
#include <limits>
#include "fft.cu"

int main()
{
    using clock = std::chrono::high_resolution_clock;
    using dur_ms = std::chrono::duration<double, std::milli>;

    std::cout << std::left
        << std::setw(12) << "N"
        //<< std::setw(15) << "DFT_CPU(ms)"
        << std::setw(15) << "FFT_CPU(ms)"
        << std::setw(15) << "FFT_GPU(ms)" << '\n';

    for (int exp = 3; exp <= 27; ++exp) 
    {
        std::size_t N = 1ULL << exp;

        std::vector<std::complex<float>> x(N), Xdft(N), XfftCpu(N), XfftGpu(N);

        /* simple (but non‑trivial) test signal */
        for (std::size_t i = 0; i < N; ++i)
            x[i] = { std::sin(0.017453292f * static_cast<float>(i)), 
                     std::cos(0.031415927f * static_cast<float>(i)) };

        ///* ---------------- CPU – DFT (brute‑force) ---------------- */
        //double t_dft = std::numeric_limits<double>::quiet_NaN();
        //if (N <= (1U << 16))                   // practical safeguard – comment out to force full range
        //{
        //    auto t0 = clock::now();
        //    dft_cpu(x.data(), Xdft.data(), N);
        //    auto t1 = clock::now();
        //    t_dft = dur_ms(t1 - t0).count();
        //}

        /* ---------------- CPU – FFT ---------------- */
        auto t0_fft_cpu = clock::now();
        fft_cpu(x.data(), XfftCpu.data(), N);
        auto t1_fft_cpu = clock::now();
        double t_fft_cpu = dur_ms(t1_fft_cpu - t0_fft_cpu).count();

        /* ---------------- GPU – FFT ---------------- */
        auto t0_fft_gpu = clock::now();
        fft_gpu(x.data(), XfftGpu.data(), N);
        cudaDeviceSynchronize();              
        auto t1_fft_gpu = clock::now();
        double t_fft_gpu = dur_ms(t1_fft_gpu - t0_fft_gpu).count();

        /* ---------------- print results ---------------- */
        std::cout << std::setw(12) << N;

        //if (std::isnan(t_dft))
        //    std::cout << std::setw(15) << "N/A";
        //else
        //    std::cout << std::setw(15) << std::fixed << std::setprecision(3) << t_dft;

        std::cout << std::setw(15) << std::fixed << std::setprecision(3) << t_fft_cpu
            << std::setw(15) << std::fixed << std::setprecision(3) << t_fft_gpu
            << '\n';
    }
    return 0;
}