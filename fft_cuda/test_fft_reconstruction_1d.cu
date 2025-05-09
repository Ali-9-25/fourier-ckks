#include <iostream>
#include <random>
#include <vector>
#include <complex>
#include <cmath>              // std::abs
#include "fft.cu"

int main()
{
    constexpr std::size_t N = 1 << 20;          // 1 048 576-point signal

    /* ---------- make a repeatable random complex signal ---------- */
    std::mt19937                 rng(2025);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<std::complex<float>> x(N), X(N), y(N);
    for (auto& v : x) v = { dist(rng), dist(rng) };

    /* ---------- FFT-> IFFT on the GPU ---------------------------- */
    fft_gpu (x.data(), X.data(), N);
    ifft_gpu(X.data(), y.data(), N);

    /* ---------- error check -------------------------------------- */
    float max_err = 0.f, rms = 0.f;
    for (std::size_t i = 0; i < N; ++i) {
        float e = std::abs(y[i] - x[i]);
        max_err = std::max(max_err, e);
        rms     += e * e;
    }
    rms = std::sqrt(rms / static_cast<float>(N));

    std::cout << "1-D reconstruction:   N = " << N << '\n'
              << "    max |error|  = " << max_err << '\n'
              << "    RMS error   = " << rms     << '\n';

    if (max_err < 1e-3f)
        std::cout << ">>> PASS\n";
    else
        std::cout << ">>> FAIL\n";

    return 0;
}
