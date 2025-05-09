/**********************************************************************
  Polynomial-multiplication performance comparison: CPU vs GPU
  Usage:  test_perf <max_log2N>
         Runs N = 2, 4, 8, …, 2^max_log2N.
  Prints a table: N, CPU(ms), GPU(ms)
 *********************************************************************/

 #include <iostream>
 #include <vector>
 #include <random>
 #include <cstdint>
 #include <chrono>
 #include <iomanip>
 #include <string>
 #include "ntt.cu"
 
 constexpr uint32_t MOD        = 998244353u;   // 119·2^23 + 1
 constexpr uint32_t ROOT       = 3u;           // primitive 2^23-rd root
 constexpr uint32_t ROOT_INV   = 332748118u;   // inverse of ROOT
 
 /* ------------------------------------------------------------------ */
 /* naïve O(N²) negacyclic convolution – CPU baseline                  */
 static void poly_mul_cpu(const std::vector<uint32_t>& A,
                          const std::vector<uint32_t>& B,
                          std::vector<uint32_t>&       C,
                          uint32_t                     p)
 {
     std::size_t N = A.size();
     C.assign(N, 0);
 
     for (std::size_t i = 0; i < N; ++i) {
         for (std::size_t j = 0; j < N; ++j) {
             uint32_t prod = static_cast<uint64_t>(A[i]) * B[j] % p;
             std::size_t idx = i + j;
             bool        wrap = (idx >= N);
             std::size_t k   = wrap ? (idx - N) : idx;
 
             uint32_t s = wrap
                         ? C[k] + (p - prod)
                         : C[k] + prod;
             C[k] = (s >= p ? s - p : s);
         }
     }
 }
 
 int main(int argc, char** argv)
 {
     if (argc < 2) {
         std::cerr << "Usage: " << argv[0] << " <max_log2N>\n";
         return 1;
     }
     unsigned max_log2 = std::stoi(argv[1]);
     std::mt19937 rng(2025);
     std::uniform_int_distribution<uint32_t> dist(0, MOD - 1);
 
     std::cout << std::setw(10) << "N"
               << std::setw(15) << "CPU (ms)"
               << std::setw(15) << "GPU (ms)"
               << "\n";
     std::cout << std::string(40, '-') << "\n";
 
     for (unsigned log2N = 1; log2N <= max_log2; ++log2N) {
         std::size_t N = 1u << log2N;
 
         // generate random inputs
         std::vector<uint32_t> A(N), B(N), C_cpu(N), C_gpu(N);
         for (auto &v : A) v = dist(rng);
         for (auto &v : B) v = dist(rng);
 
         // warm up GPU once
         poly_mul_gpu(A.data(), B.data(), C_gpu.data(),
                      N, ROOT, ROOT_INV, MOD);
         cudaDeviceSynchronize();
 
         // --- CPU timing ---
         auto t0 = std::chrono::high_resolution_clock::now();
         poly_mul_cpu(A, B, C_cpu, MOD);
         auto t1 = std::chrono::high_resolution_clock::now();
         double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
 
         // --- GPU timing with CUDA events ---
         cudaEvent_t start, stop;
         cudaEventCreate(&start);
         cudaEventCreate(&stop);
         cudaEventRecord(start, 0);
 
         poly_mul_gpu(A.data(), B.data(), C_gpu.data(),
                      N, ROOT, ROOT_INV, MOD);
 
         cudaEventRecord(stop, 0);
         cudaEventSynchronize(stop);
         float gpu_ms = 0;
         cudaEventElapsedTime(&gpu_ms, start, stop);
 
         cudaEventDestroy(start);
         cudaEventDestroy(stop);
 
         // optional correctness check (can disable for pure perf runs)
         // for (std::size_t i = 0; i < N; ++i)
         //     if (C_cpu[i] != C_gpu[i])
         //         std::cerr << "Error at " << i << "\n";
 
         std::cout << std::setw(10) << N
                   << std::setw(15) << std::fixed << std::setprecision(3) << cpu_ms
                   << std::setw(15) << std::fixed << std::setprecision(3) << gpu_ms
                   << "\n";
     }
     return 0;
 }
 