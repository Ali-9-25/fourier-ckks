#include "ntt.cu"

constexpr uint32_t MOD  = 998244353u;
constexpr uint32_t ROOT = 3u;                   
constexpr uint32_t ROOT_INV = 332748118u;        
constexpr uint32_t MOD_MINUS1 = MOD - 1;

int main()
{
    constexpr std::size_t N = 8;
    std::vector<uint32_t> x(N), Xcpu(N), Xgpu(N);
    for (size_t i = 0; i < N; ++i) x[i] = x[i] = (i * i + 7) % MOD;
    uint32_t N_inverse = mod_pow(static_cast<uint32_t>(N), MOD_MINUS1 - 1, MOD);

    ntt_cpu (x.data(), Xcpu.data(), N, N_inverse, ROOT, MOD);
    ntt_gpu(x.data(), Xgpu.data(), N, N_inverse, ROOT, MOD);

    std::cout << "k    CPU\t\t\tGPU\n";
    for (size_t k = 0; k < N; ++k) {
        std::cout << k << ":  "
            << Xcpu[k] << "   "
            << Xgpu[k] << "\n";
    }
}
