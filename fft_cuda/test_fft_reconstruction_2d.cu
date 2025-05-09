#include <iostream>
#include <random>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include "fft.cu"

int main()
{
    constexpr std::size_t ROWS = 512, COLS = 512;    // 512×512 image
    constexpr std::size_t TOTAL = ROWS * COLS;

    std::mt19937 rng(2048);
    std::normal_distribution<float> dist(0.f, 0.5f);

    std::vector<std::complex<float>> img(TOTAL), F(TOTAL), rec(TOTAL);
    for (auto& v : img) v = { dist(rng), dist(rng) };

    /* --------------- forward & inverse 2-D FFT ------------------- */
    fft2d_gpu (img.data(), F.data(),   ROWS, COLS);
    ifft2d_gpu(F.data(),  rec.data(), ROWS, COLS);

    /* --------------- error metrics ------------------------------- */
    double max_err = 0.0, mse = 0.0;
    for (std::size_t i = 0; i < TOTAL; ++i) {
        double e = std::abs(rec[i] - img[i]);
        max_err = std::max(max_err, e);
        mse    += e * e;
    }
    mse /= static_cast<double>(TOTAL);
    double rmse = std::sqrt(mse);

    std::cout << "2-D reconstruction:  " << ROWS << "×" << COLS << '\n'
              << "    max |error| = " << max_err << '\n'
              << "    RMSE       = " << rmse    << '\n';

    if (max_err < 1e-3)
        std::cout << ">>> PASS\n";
    else
        std::cout << ">>> FAIL\n";

    return 0;
}
