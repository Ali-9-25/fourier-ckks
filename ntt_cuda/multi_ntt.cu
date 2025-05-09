#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <iostream>

/* ---------- helpers: modular arithmetic -------------------------- */
__host__ __device__ static inline
uint32_t mod_add(uint32_t a, uint32_t b, uint32_t p) {
    uint32_t s = a + b;
    return (s >= p ? s - p : s);
}
__host__ __device__ static inline
uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t p) {
    return (a >= b ? a - b : a + p - b);
}
__host__ __device__ static inline
uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t p) {
    return static_cast<uint64_t>(a) * b % p;
}

__host__ __device__ static inline
uint32_t mod_pow(uint32_t a, uint64_t e, uint32_t p) {
    uint32_t res = 1;
    while (e) {
        if (e & 1) res = mod_mul(res, a, p);
        a = mod_mul(a, a, p);
        e >>= 1;
    }
    return res;
}

/* ---------- bit reversal ------------ */
__host__ __device__
unsigned bit_reverse(unsigned v, int lgN)
{
    v = ((v & 0x55555555u) << 1) | ((v & 0xAAAAAAAAu) >> 1);
    v = ((v & 0x33333333u) << 2) | ((v & 0xCCCCCCCCu) >> 2);
    v = ((v & 0x0F0F0F0Fu) << 4) | ((v & 0xF0F0F0F0u) >> 4);
    v = ((v & 0x00FF00FFu) << 8) | ((v & 0xFF00FF00u) >> 8);
    v = (v << 16) | (v >> 16);
    return v >> (32 - lgN);
}

__global__ void bit_reverse_multi_kernel(const uint32_t* in, uint32_t* out, int lgN, uint32_t N, uint32_t num_of_polynomials)
{
    unsigned tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * num_of_polynomials) return;
    out[tid] = in[((tid / N) * N) + bit_reverse(tid % N, lgN)];
}

__global__ void scale_multi_kernel(uint32_t* data, uint32_t N, uint32_t* N_inverses, uint32_t* primes, uint32_t num_of_polynomials)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * num_of_polynomials) return;
    uint32_t poly_idx = tid / N;
    data[tid] = mod_mul(data[tid], N_inverses[poly_idx], primes[poly_idx]);
}

__global__ void roots_unity_kernel(uint32_t* primitive_roots, uint32_t* w_roots_unity, uint32_t* primes, uint32_t num_of_primes, uint32_t m)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_of_primes) return;
    uint32_t p = primes[tid];
    w_roots_unity[tid] = mod_pow(primitive_roots[tid], (p - 1) / m, p);
}

__global__ void N_inverses_kernel(uint32_t N, uint32_t* primes, uint32_t num_of_primes, uint32_t* N_inverses)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_of_primes) return;
    uint32_t p = primes[tid];
    N_inverses[tid] =  mod_pow(N, p - 2, p);
}

__global__ void ntt_stage_multi_kernel(uint32_t*  data, uint32_t N, uint32_t* w_roots_unity, uint32_t* primes, uint32_t stage, uint32_t num_of_polynomials, uint32_t num_of_primes)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * num_of_polynomials / 2) return;
    uint32_t halfN = N >> 1;
    uint32_t poly_idx  = tid / halfN;
    uint32_t local_tid  = tid % halfN;

    uint32_t mh = 1u << (stage - 1);   
    uint32_t j  =  local_tid % mh;

    uint32_t idx1 = poly_idx * N + ((local_tid / mh) * (mh << 1)) + j;
    uint32_t idx2 = idx1 + mh;

    uint32_t u = data[idx1];
    uint32_t params_idx = poly_idx >= num_of_primes ? poly_idx - num_of_primes : poly_idx;
    uint32_t p = primes[params_idx];
    uint32_t t = mod_mul(mod_pow(w_roots_unity[params_idx], j, p), data[idx2], p);


    data[idx1] = mod_add(u, t, p);
    data[idx2] = mod_sub(u, t, p);
}

static int ntt_multi(const uint32_t* input_polynomials, uint32_t* output_polynomials, uint32_t N, uint32_t* N_inverses, uint32_t* primitive_roots, uint32_t* primes, uint32_t num_of_primes, uint32_t num_of_polynomials)
{
    if (N == 0 || (N & (N - 1))) return -1;

    const size_t bytes  = num_of_polynomials * N * sizeof(uint32_t);

    uint32_t* d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, input_polynomials, bytes, cudaMemcpyHostToDevice);

    // one bit-reverse for both
    int lgN = static_cast<int>(std::log2(N));
    dim3 thr(256), blkMulti((num_of_polynomials * N + 255) / 256);
    uint32_t* d_tmp;
    cudaMalloc(&d_tmp, bytes);

    bit_reverse_multi_kernel<<<blkMulti, thr>>>(d_data, d_tmp, lgN, N, num_of_polynomials);
    cudaDeviceSynchronize();
    cudaFree(d_data);
    d_data = d_tmp;

    // iterative NTT stages (unchanged)
    dim3 blkMultiHalf((num_of_polynomials * N / 2 + 255) / 256);
    dim3 blkMultiPrimes((num_of_primes + 255) / 256);
    uint32_t* w_roots_unity;
    cudaMalloc(&w_roots_unity, num_of_primes * sizeof(uint32_t));
    for (std::size_t m = 2, stage = 1; m <= N; m <<= 1, stage++) {
        roots_unity_kernel<<<blkMultiPrimes, thr>>>(primitive_roots, w_roots_unity, primes, num_of_primes, m);
        cudaDeviceSynchronize();
        ntt_stage_multi_kernel<<<blkMultiHalf, thr>>>(d_data, N, w_roots_unity, primes, stage, num_of_polynomials, num_of_primes);
        cudaDeviceSynchronize();
    }

    // one scale for both if inverse
    if (N_inverses != nullptr) {
        scale_multi_kernel<<<blkMulti, thr>>>(d_data, N, N_inverses, primes, num_of_polynomials);
        cudaDeviceSynchronize();
    }

    // copy-back
    cudaMemcpy(output_polynomials, d_data, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    return 0;
}

__global__ void precompute_psi_kernel(uint32_t* psi, uint32_t* psi_inv, uint32_t* g, uint32_t* p, uint32_t N, uint32_t num_polys) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_polys * N) return;

    uint32_t poly_idx = tid / N;
    uint32_t idx = tid % N;

    uint32_t prime = p[poly_idx];
    uint32_t generator = g[poly_idx];

    uint32_t exp = (prime - 1) / (2 * N);
    uint32_t psi_base = mod_pow(generator, exp, prime);
    uint32_t psi_base_inv = mod_pow(psi_base, prime - 2, prime);

    psi[tid] = mod_pow(psi_base, idx, prime);
    psi_inv[tid] = mod_pow(psi_base_inv, idx, prime);
}


__global__ void twist_kernel(uint32_t* poly, uint32_t* psi, uint32_t* primes, uint32_t N, uint32_t num_polys) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * num_polys) return;

    uint32_t poly_idx = tid / N;
    uint32_t prime = primes[poly_idx];

    poly[tid] = mod_mul(poly[tid], psi[tid], prime);
}

__global__ void untwist_kernel(uint32_t* poly, uint32_t* psi_inv, uint32_t* primes, uint32_t N, uint32_t num_polys) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * num_polys) return;

    uint32_t poly_idx = tid / N;
    uint32_t prime = primes[poly_idx];

    poly[tid] = mod_mul(poly[tid], psi_inv[tid], prime);
}

__global__ void poly_mul_pointwise_kernel(uint32_t* A_ntt, uint32_t* B_ntt, uint32_t* C_ntt, uint32_t* primes, uint32_t N, uint32_t num_polys) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * num_polys) return;

    uint32_t poly_idx = tid / N;
    uint32_t prime = primes[poly_idx];

    C_ntt[tid] = mod_mul(A_ntt[tid], B_ntt[tid], prime);
}

int poly_mul_multi_gpu(const uint32_t* A_host,
                       const uint32_t* B_host,
                       uint32_t* C_host,
                       uint32_t N,
                       uint32_t* primitive_roots,
                       uint32_t* primitive_roots_inv,
                       uint32_t* primes,
                       uint32_t num_polys) {

    size_t poly_bytes = num_polys * N * sizeof(uint32_t);

    uint32_t *d_inputs, *d_C, *psi, *psi_inv;
    cudaMalloc(&d_inputs, 2 * poly_bytes);
    cudaMalloc(&d_C, poly_bytes);
    cudaMalloc(&psi, poly_bytes);
    cudaMalloc(&psi_inv, poly_bytes);

    cudaMemcpy(d_inputs, A_host, poly_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs + num_polys * N, B_host, poly_bytes, cudaMemcpyHostToDevice);

    uint32_t *d_primitive_roots, *d_primitive_roots_inv, *d_primes, *d_A, *d_B;
    d_A = d_inputs;
    d_B = d_inputs + num_polys * N;
    cudaMalloc(&d_primitive_roots, num_polys * sizeof(uint32_t));
    cudaMalloc(&d_primitive_roots_inv, num_polys * sizeof(uint32_t));
    cudaMalloc(&d_primes, num_polys * sizeof(uint32_t));
    cudaMemcpy(d_primitive_roots, primitive_roots, num_polys * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_primitive_roots_inv, primitive_roots_inv, num_polys * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_primes, primes, num_polys * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Precompute psi
    dim3 threads(256);
    dim3 blocks_poly((num_polys * N + 255) / 256);  // FIXED HERE
    precompute_psi_kernel<<<blocks_poly, threads>>>(psi, psi_inv, d_primitive_roots, d_primes, N, num_polys);
    cudaDeviceSynchronize();

    // Pre-twist
    dim3 blocks_data((num_polys * N + 255) / 256);
    twist_kernel<<<blocks_data, threads>>>(d_A, psi, d_primes, N, num_polys);
    twist_kernel<<<blocks_data, threads>>>(d_B, psi, d_primes, N, num_polys);
    cudaDeviceSynchronize();

    // Forward NTT on A
    ntt_multi(d_inputs, d_inputs, N, nullptr, d_primitive_roots, d_primes, num_polys, 2 * num_polys);

    // Pointwise multiply
    poly_mul_pointwise_kernel<<<blocks_data, threads>>>(d_A, d_B, d_C, d_primes, N, num_polys);
    cudaDeviceSynchronize();

    // Compute inverses
    uint32_t* d_N_inv;
    cudaMalloc(&d_N_inv, num_polys*sizeof(uint32_t));
    dim3 blocks_inv((num_polys+255)/256);
    N_inverses_kernel<<<blocks_inv, threads>>>(N, d_primes, num_polys, d_N_inv);
    cudaDeviceSynchronize();

    // Inverse NTT
    ntt_multi(d_C, d_C, N, d_N_inv, d_primitive_roots_inv, d_primes, num_polys, num_polys);
    cudaDeviceSynchronize();

    // Post-twist
    untwist_kernel<<<blocks_data, threads>>>(d_C, psi_inv, d_primes, N, num_polys);
    cudaDeviceSynchronize();
    cudaMemcpy(C_host, d_C, poly_bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(psi);
    cudaFree(psi_inv);
    cudaFree(d_N_inv);

    return 0;
}

