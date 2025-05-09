/**********************************************************************
  Polynomial-multiplication demo & correctness check
  Usage:  poly_mul_test <log2N> [print]
          <log2N>  – power of two for the transform length (N = 2^log2N)
          [print]  – any second arg prints A(x), B(x), C(x) to stdout
 *********************************************************************/

 #include <iostream>
 #include <vector>
 #include <random>
 #include <cstdint>
 #include <cstring>
 #include <string>
 #include "ntt.cu"
 
 constexpr uint32_t MOD        = 998244353u;   // 119 · 2^23 + 1
 constexpr uint32_t ROOT       = 3u;           // primitive 2^23-rd root
 constexpr uint32_t ROOT_INV   = 332748118u;
 
/* ------------------------------------------------------------------ */
/* naïve O(N²) **negacyclic** convolution – CPU baseline              */
static void poly_mul_cpu(const std::vector<uint32_t>& A,
    const std::vector<uint32_t>& B,
    std::vector<uint32_t>&       C,
    uint32_t                     p)
{
    std::size_t N = A.size();
    C.assign(N, 0);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            // compute raw product mod p
            uint32_t prod = static_cast<uint64_t>(A[i]) * B[j] % p;

            // negacyclic index: wrap around with a sign-flip whenever i+j >= N
            std::size_t idx = i + j;
            bool        wrap = (idx >= N);
            std::size_t k   = wrap ? (idx - N) : idx;

            // add or subtract prod into C[k]
            uint32_t s;
            if (!wrap) {
                // normal addition
                s = C[k] + prod;
            } else {
                // subtraction = add (p - prod)
                s = C[k] + (p - prod);
            }
            // reduce mod p
            C[k] = (s >= p ? s - p : s);
        }
    }
}

 
 /* ------------------------------------------------------------------ */
 static std::string poly_to_string(const std::vector<uint32_t>& a)
 {
     const char* xpow[] = { "", "x", "x^2", "x^3", "x^4", "x^5",
                            "x^6", "x^7", "x^8", "x^9" };   // up to 2^10-1
     std::string s;
     int n = static_cast<int>(a.size());
     for (int i = n - 1; i >= 0; --i) {
         if (a[i] == 0) continue;
         if (!s.empty()) s += " + ";
         s += std::to_string(a[i]);
         if (i)
             s += (i < 10 ? xpow[i] : "x^" + std::to_string(i));
     }
     if (s.empty()) s = "0";
     return s;
 }
 
 /* ------------------------------------------------------------------ */
 int main(int argc, char** argv)
 {
     if (argc < 2) {
         std::cerr << "Usage: " << argv[0] << " <log2N> [print]\n"; return 1;
     }
     unsigned log2N = std::stoi(argv[1]);
     const std::size_t N = 1u << log2N;
 
     /* ---- random test polynomials over Z_p ----------------------------- */
     std::mt19937 rng(2025);
     std::uniform_int_distribution<uint32_t> dist(0, MOD - 1);
 
     std::vector<uint32_t> A(N), B(N), C_gpu(N), C_cpu;
     for (auto& v : A) v = dist(rng);
     for (auto& v : B) v = dist(rng);
 
     /* ---- GPU polynomial multiplication -------------------------------- */
     if ( poly_mul_gpu(A.data(), B.data(), C_gpu.data(),
                       N, ROOT, ROOT_INV, MOD) ) {
         std::cerr << "GPU routine returned error\n"; return 1;
     }
 
     /* ---- CPU reference ------------------------------------------------- */
     poly_mul_cpu(A, B, C_cpu, MOD);
 
     /* ---- correctness check -------------------------------------------- */
     for (std::size_t i = 0; i < N; ++i)
         if (C_cpu[i] != C_gpu[i]) {
             std::cerr << "Mismatch at " << i << ": "
                       << C_cpu[i] << " (CPU)  vs  "
                       << C_gpu[i] << " (GPU)\n";
            return 1;
         }
 
     std::cout << ">>> PASS (N = " << N << ")\n";
 
     /* ---- optional pretty print ---------------------------------------- */
     if (argc > 2) {
         std::cout << "\nA(x) = " << poly_to_string(A) << "\n\n";
         std::cout << "B(x) = " << poly_to_string(B) << "\n\n";
         std::cout << "C(x) = " << poly_to_string(C_gpu) << "\n";
     }
     return 0;
 }
 