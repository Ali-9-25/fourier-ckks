#include <iostream>
#include <random>
#include <vector>
#include <complex>
#include <cmath>          
#include "ntt.cu"

constexpr uint32_t MOD  = 998244353u;
constexpr uint32_t ROOT = 3u;                   
constexpr uint32_t ROOT_INV = 332748118u;        
constexpr uint32_t MOD_MINUS1 = MOD - 1;

int main()
{
    constexpr std::size_t N = 1 << 20;          // 1 048 576-point signal

    /* ---------- make a repeatable random complex signal ---------- */
    std::mt19937                 rng(2025);
    std::uniform_int_distribution<uint32_t> dist(0, MOD_MINUS1);
    uint32_t N_inverse = mod_pow(static_cast<uint32_t>(N), MOD_MINUS1 - 1, MOD);

    std::vector<uint32_t> x(N), X(N), y(N);
    for (auto& v : x) v = dist(rng);

    /* ---------- FFT-> IFFT on the GPU ---------------------------- */
    ntt_gpu(x.data(), X.data(), N, N_inverse, ROOT, MOD);
    intt_gpu(X.data(), y.data(), N, N_inverse, ROOT_INV, MOD);

    /* ---------- error check -------------------------------------- */
    std::cout << "Error Checking:\n";
    for (std::size_t i = 0; i < N; ++i)
        if (x[i] != y[i]) {
            std::cout << "Error at index " << i << ": " << x[i] << " != " << y[i] << "\n";
            return 1;
        }

    std::cout << ">>> PASS\n";
    return 0;
}
