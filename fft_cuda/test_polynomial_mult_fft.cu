/* ==================================================================== *
     Polynomial multiplication:  Naïve CPU O(N²)   vs.   GPU FFT O(N log N)
     --------------------------------------------------------------------
     Usage:
         nvcc -O3 test_polynomial_mult_fft.cu -o poly_fft
         ./poly_fft [exp]
       where exp is the power‑of‑two such that  deg = 2^exp – 1.
       Example:  ./poly_fft 15     -> degree 32767 × 32767
 * ==================================================================== */

 #include <cuda_runtime.h>
 #include <cuComplex.h>
 
 #include <algorithm>
 #include <cassert>
 #include <chrono>
 #include <cmath>
 #include <iomanip>
 #include <iostream>
 #include <random>
 #include <vector>
 
 #include "fft.cu"          /* FFT helpers + pointwise_mul_kernel       */
 
 /* ---------- naïve reference multiplication (CPU, 64‑bit result) ---- */
 static void poly_mul_cpu(const std::vector<int>&  a,
                          const std::vector<int>&  b,
                          std::vector<long long>&  c)
 {
     std::size_t na = a.size(), nb = b.size();
     c.assign(na + nb - 1, 0LL);
     for (std::size_t i = 0; i < na; ++i)
         for (std::size_t j = 0; j < nb; ++j)
             c[i + j] += static_cast<long long>(a[i]) * b[j];
 }
 
 /* ---------- helpers ------------------------------------------------- */
 static inline long long lround_ll(float x)
 { return static_cast<long long>(std::llround(static_cast<double>(x))); }
 
 static std::size_t next_pow2(std::size_t v)
 {
     if (v == 0) return 1;
     --v;
     v |= v >> 1;  v |= v >> 2;  v |= v >> 4;
     v |= v >> 8;  v |= v >> 16; v |= v >> 32;
     return v + 1;
 }
 /* ==================================================================== */
 int main(int argc, char* argv[])
 {
     /* --- choose polynomial degrees -------------------------------- */
     int    exp  = (argc >= 2 ? std::stoi(argv[1]) : 14);   // default 2^14‑1
     size_t degA = (1uLL << exp) - 1;
     size_t degB = degA;                                    // same length
 
     std::cout << "Polynomial multiplication (deg = " << degA
               << ") using GPU FFT vs. naive CPU\n";
 
     /* --- construct random integer polynomials --------------------- */
     std::mt19937                      rng(2025);
     std::uniform_int_distribution<int> dist(-100, 100);
 
     std::vector<int> a(degA + 1), b(degB + 1);
     for (int& v : a) v = dist(rng);
     for (int& v : b) v = dist(rng);
 
     /* --- reference result on CPU ---------------------------------- */
     std::vector<long long> ref;
     auto t0_cpu = std::chrono::high_resolution_clock::now();
     poly_mul_cpu(a, b, ref);
     auto t1_cpu = std::chrono::high_resolution_clock::now();
     double cpu_ms =
         std::chrono::duration<double, std::milli>(t1_cpu - t0_cpu).count();
 
     /* --- prepare zero‑padded complex arrays for FFT --------------- */
     size_t nConv = ref.size();
     size_t Nfft  = next_pow2(nConv);
 
     std::vector<std::complex<float>>
         A(Nfft), B(Nfft), FA(Nfft), FB(Nfft),
         FC(Nfft), C_time(Nfft);
 
     for (size_t i = 0; i < a.size(); ++i) A[i] = { static_cast<float>(a[i]), 0.f };
     for (size_t i = 0; i < b.size(); ++i) B[i] = { static_cast<float>(b[i]), 0.f };
 
     /* ---------------- GPU pipeline -------------------------------- */
 
     auto t0_gpu = std::chrono::high_resolution_clock::now();
 
     /* 1) forward FFTs */
     fft_gpu(A.data(), FA.data(), Nfft);
     fft_gpu(B.data(), FB.data(), Nfft);
 
     /* 2) element‑wise (Hadamard) product */
     cuFloatComplex *dFA{}, *dFB{}, *dFC{};
     size_t bytes = Nfft * sizeof(cuFloatComplex);
     cudaMalloc(&dFA, bytes);
     cudaMalloc(&dFB, bytes);
     cudaMalloc(&dFC, bytes);
 
     cudaMemcpy(dFA, FA.data(), bytes, cudaMemcpyHostToDevice);
     cudaMemcpy(dFB, FB.data(), bytes, cudaMemcpyHostToDevice);
 
     dim3 threads(256);
     dim3 blocks((Nfft + threads.x - 1) / threads.x);
     pointwise_mul_kernel<<<blocks, threads>>>(dFA, dFB, dFC, Nfft);
 
     cudaMemcpy(FC.data(), dFC, bytes, cudaMemcpyDeviceToHost);
     cudaFree(dFA); cudaFree(dFB); cudaFree(dFC);
 
     /* 3) inverse FFT */
     ifft_gpu(FC.data(), C_time.data(), Nfft);
 
     cudaDeviceSynchronize();            // ensure all GPU work finished
     auto t1_gpu = std::chrono::high_resolution_clock::now();
     double gpu_ms =
         std::chrono::duration<double, std::milli>(t1_gpu - t0_gpu).count();
 
     /* --- round & compare ------------------------------------------ */
     std::vector<long long> result(ref.size());
     for (size_t i = 0; i < ref.size(); ++i)
         result[i] = lround_ll(C_time[i].real());
 
     long long max_abs_err = 0;
     for (size_t i = 0; i < ref.size(); ++i)
         max_abs_err = std::max(max_abs_err,
                                std::llabs(ref[i] - result[i]));
 
     /* ---------------- report -------------------------------------- */
     std::cout << std::fixed << std::setprecision(3)
               << "    convolution length      : " << nConv      << '\n'
               << "    FFT size (power of two) : " << Nfft       << '\n'
               << "    CPU naive time          : " << cpu_ms << " ms\n"
               << "    GPU FFT   time          : " << gpu_ms << " ms\n"
               << "    Speed up (GPU / CPU)    : " << (cpu_ms / gpu_ms) << "x\n"
               << "    max |error|             : " << max_abs_err << '\n';
 
     if (max_abs_err == 0)
         std::cout << ">>> PASS  - exact match\n";
     else if (max_abs_err <= 1)
         std::cout << ">>> PASS  - round-off <= 1\n";
     else
         std::cout << ">>> FAIL  - error (" << max_abs_err << ") too large\n";
 
     return 0;
 }
 