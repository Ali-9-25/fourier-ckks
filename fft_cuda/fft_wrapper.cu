// fft_wrapper.cu
#include "fft.cu"           // assumes fft.cu is in same directory
#include <cstdlib>
#include <vector>
#include <complex>

extern "C" {

// 1D forward FFT: in_real/in_imag → out_real/out_imag
__declspec(dllexport)
int fft1d_gpu_c(const float* in_real, const float* in_imag,
                float*       out_real, float*       out_imag,
                unsigned     N)
{
    std::vector<std::complex<float>> in(N), out(N);
    for (unsigned i = 0; i < N; ++i)
        in[i] = std::complex<float>(in_real[i], in_imag[i]);
    int err = fft_gpu(in.data(), out.data(), N);
    if (err) return err;
    for (unsigned i = 0; i < N; ++i) {
        out_real[i] = out[i].real();
        out_imag[i] = out[i].imag();
    }
    return 0;
}

// 1D inverse FFT
__declspec(dllexport)
int ifft1d_gpu_c(const float* in_real, const float* in_imag,
                 float*       out_real, float*       out_imag,
                 unsigned     N)
{
    std::vector<std::complex<float>> in(N), out(N);
    for (unsigned i = 0; i < N; ++i)
        in[i] = std::complex<float>(in_real[i], in_imag[i]);
    int err = ifft_gpu(in.data(), out.data(), N);
    if (err) return err;
    for (unsigned i = 0; i < N; ++i) {
        out_real[i] = out[i].real();
        out_imag[i] = out[i].imag();
    }
    return 0;
}

// 2D forward FFT: inputs flattened row-major
__declspec(dllexport)
int fft2d_gpu_c(const float* in_real, const float* in_imag,
                float*       out_real, float*       out_imag,
                unsigned     rows, unsigned cols)
{
    size_t total = (size_t)rows * cols;
    std::vector<std::complex<float>> in(total), out(total);
    for (size_t i = 0; i < total; ++i)
        in[i] = std::complex<float>(in_real[i], in_imag[i]);
    int err = fft2d_gpu(in.data(), out.data(), rows, cols);
    if (err) return err;
    for (size_t i = 0; i < total; ++i) {
        out_real[i] = out[i].real();
        out_imag[i] = out[i].imag();
    }
    return 0;
}

// 2D inverse FFT
__declspec(dllexport)
int ifft2d_gpu_c(const float* in_real, const float* in_imag,
                 float*       out_real, float*       out_imag,
                 unsigned     rows, unsigned cols)
{
    size_t total = (size_t)rows * cols;
    std::vector<std::complex<float>> in(total), out(total);
    for (size_t i = 0; i < total; ++i)
        in[i] = std::complex<float>(in_real[i], in_imag[i]);
    int err = ifft2d_gpu(in.data(), out.data(), rows, cols);
    if (err) return err;
    for (size_t i = 0; i < total; ++i) {
        out_real[i] = out[i].real();
        out_imag[i] = out[i].imag();
    }
    return 0;
}

} // extern "C"
