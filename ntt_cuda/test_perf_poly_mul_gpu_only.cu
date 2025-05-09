/**********************************************************************
  GPU-only polynomial-multiply timing (negacyclic via NTT)
  Usage:  test_gpu_only <max_log2N>
         Measures poly_mul_gpu for N = 2, 4, 8, …, 2^max_log2N.
 *********************************************************************/

 #include <iostream>
 #include <vector>
 #include <random>
 #include <cstdint>
 #include <iomanip>
 #include <cuda_runtime.h>
 #include <string>
 #include "ntt.cu"
 
 constexpr uint32_t MOD        = 998244353u;   // 119·2^23 + 1
 constexpr uint32_t ROOT       = 3u;           // primitive 2^23-rd root
 constexpr uint32_t ROOT_INV   = 332748118u;   // inverse of ROOT
 
 int main(int argc, char** argv) {
     if (argc < 2) {
         std::cerr << "Usage: " << argv[0] << " <max_log2N>\n";
         return 1;
     }
     unsigned max_log2 = std::stoi(argv[1]);
     std::mt19937 rng(2025);
     std::uniform_int_distribution<uint32_t> dist(0, MOD - 1);
 
     // Header
     std::cout << std::setw(10) << "N"
               << std::setw(15) << "GPU-only (ms)"
               << "\n"
               << std::string(25, '-') << "\n";
 
     // Loop over sizes
     for (unsigned log2N = 1; log2N <= max_log2; ++log2N) {
         std::size_t N = 1u << log2N;
 
         // Random input polynomials
         std::vector<uint32_t> A(N), B(N), C(N);
         for (auto &v : A) v = dist(rng);
         for (auto &v : B) v = dist(rng);
 
         // warm-up
         poly_mul_gpu(A.data(), B.data(), C.data(),
                      N, ROOT, ROOT_INV, MOD);
         cudaDeviceSynchronize();
 
         // GPU timing via CUDA events
         cudaEvent_t start, stop;
         cudaEventCreate(&start);
         cudaEventCreate(&stop);
 
         cudaEventRecord(start, 0);
         poly_mul_gpu(A.data(), B.data(), C.data(),
                      N, ROOT, ROOT_INV, MOD);
         cudaEventRecord(stop, 0);
 
         cudaEventSynchronize(stop);
         float gpu_ms = 0.f;
         cudaEventElapsedTime(&gpu_ms, start, stop);
 
         cudaEventDestroy(start);
         cudaEventDestroy(stop);
 
         // Output
         std::cout << std::setw(10) << N
                   << std::setw(15) << std::fixed << std::setprecision(3) << gpu_ms
                   << "\n";
     }
 
     return 0;
 }
 