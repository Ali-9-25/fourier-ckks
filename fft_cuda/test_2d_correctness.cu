#include "fft.cu"

int main()
{
    const std::size_t ROWS = 4, COLS = 8;       
    const std::size_t TOTAL = ROWS * COLS;

    std::vector<std::complex<float>> x(TOTAL), Xcpu(TOTAL), Xgpu(TOTAL);

    for (std::size_t r = 0; r < ROWS; ++r)
        for (std::size_t c = 0; c < COLS; ++c)
            x[r * COLS + c] = { static_cast<float>(r + 1), static_cast<float>(c + 1) };

    fft2d_cpu(x.data(), Xcpu.data(), ROWS, COLS);
    fft2d_gpu(x.data(), Xgpu.data(), ROWS, COLS);

    auto dump = [&](const char* tag, const std::vector<std::complex<float>>& a)
        {
            std::cout << tag << "\n";
            for (std::size_t r = 0; r < ROWS; ++r)
            {
                for (std::size_t c = 0; c < COLS; ++c)
                {
                    const auto& v = a[r * COLS + c];
                    std::cout << '(' << v.real() << ',' << v.imag() << ")  ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        };

    dump("CPU 2‑D FFT:", Xcpu);
    dump("GPU 2‑D FFT:", Xgpu);

    return 0;
}
