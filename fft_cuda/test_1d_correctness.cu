#include "fft.cu"

int main()
{
    constexpr std::size_t N = 8;
    std::vector<std::complex<float>> x(N), Xcpu(N), Xgpu(N);
    for (size_t i = 0; i < N; ++i) x[i] = std::complex<float>(i + 1, i % 3);

    fft_cpu(x.data(), Xcpu.data(), N);
    fft_gpu(x.data(), Xgpu.data(), N);

    std::cout << "k    CPU\t\t\tGPU\n";
    for (size_t k = 0; k < N; ++k) {
        std::cout << k << ":  "
            << Xcpu[k] << "   "
            << Xgpu[k] << "\n";
    }
}