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

/* ================================================================== *
                            CPU NTT
* ================================================================== */
static int ntt_cpu_generic(const uint32_t* in,
                        uint32_t*       out,
                        std::size_t     N,
                        std::size_t N_inverse,
                        uint32_t w_primitive_root,
                        uint32_t p)
{
    if (N == 0 || (N & (N - 1))) return -1; 

    int lgN = static_cast<int>(std::log2(N));

    for (std::size_t i = 0; i < N; ++i)
        out[bit_reverse(i, lgN)] = in[i];

    for (std::size_t m = 2, stage = 1; m <= N; m <<= 1, ++stage) {
        std::size_t mh = m >> 1;
        uint32_t w_root_unity = mod_pow(w_primitive_root, (p - 1) / m, p);

        for (std::size_t k = 0; k < N; k += m) {
            uint32_t w = 1;
            for (std::size_t j = 0; j < mh; ++j) {
                uint32_t u = out[k + j];
                uint32_t v = mod_mul(out[k + j + mh], w, p);
                out[k + j]       = mod_add(u, v, p);
                out[k + j + mh]  = mod_sub(u, v, p);
                w = mod_mul(w, w_root_unity, p);
            }
        }
    }
    if (N_inverse != 0) {
        for (std::size_t i = 0; i < N; ++i)
            out[i] = mod_mul(out[i], N_inverse, p);
    }
    return 0;
}
int  ntt_cpu (const uint32_t* in, uint32_t* out, std::size_t N, std::size_t N_inverse, uint32_t w_primitive_root, uint32_t p)
{ return ntt_cpu_generic(in, out, N, 0, w_primitive_root, p); }
int  intt_cpu(const uint32_t* in, uint32_t* out, std::size_t N, std::size_t N_inverse, uint32_t w_primitive_root, uint32_t p)
{ return ntt_cpu_generic(in, out, N, N_inverse, w_primitive_root, p); }

/* ================================================================== *
                    GPU kernels (radix-2 butterfly)
* ================================================================== */ 
__global__ void bit_reverse_kernel_int(const uint32_t* in,
                                    uint32_t*       out,
                                    int             lgN)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = 1u << lgN;
    if (i >= n) return;
    out[i] = in[bit_reverse(i, lgN)];
}

__global__ void ntt_stage_kernel(uint32_t* data,
                                std::size_t N,
                                uint32_t w_root_unity,
                                uint32_t p,
                                uint32_t stage)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (N >> 1)) return;

    std::size_t mh = 1u << (stage - 1);

    std::size_t k = (tid / mh) * (mh << 1);
    std::size_t j = tid % mh;

    uint32_t u = data[k + j];
    uint32_t t = mod_mul(mod_pow(w_root_unity, j, p), data[k + j + mh], p);
    data[k + j] = mod_add(u, t, p);
    data[k + j + mh] = mod_sub(u, t, p);
}

/* ---------- scale kernel (needed for inverse) --------------------- */
__global__ void scale_kernel_int(uint32_t* data,
                                std::size_t N,
                                uint32_t    N_inverse,
                                uint32_t    p)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    data[i] = mod_mul(data[i], N_inverse, p);
}

/* ================================================================== *
                GPU host-side driver  (forward / inverse)
* ================================================================== */

static int ntt_gpu_dir(const uint32_t* host_in,
                    uint32_t*       host_out,
                    std::size_t     N,
                    std::size_t N_inverse,
                    uint32_t        w_primitive_root,
                    uint32_t        p)
{
    if (N == 0 || (N & (N - 1))) return -1;

    int    lgN   = static_cast<int>(std::log2(N));
    size_t bytes = N * sizeof(uint32_t);

    uint32_t *d_in{}, *d_data{};
    cudaMalloc(&d_in,   bytes);
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_in, host_in, bytes, cudaMemcpyHostToDevice);

    /* --- bit-reverse copy --------------------------------------- */
    dim3 threads(256), blocks((N + 255) / 256);
    bit_reverse_kernel_int<<<blocks, threads>>>(d_in, d_data, lgN);

    size_t half = N >> 1;
    dim3 blocks2((half + 255) / 256);
    /* --- iterative stages --------------------------------------- */
    for (std::size_t m = 2, stage = 1; m <= N; m <<= 1, ++stage) {
        uint32_t w_root_unity = mod_pow(w_primitive_root, (p - 1) / m, p);
        ntt_stage_kernel<<<blocks2, threads>>>(d_data, N, w_root_unity, p, stage);
        cudaDeviceSynchronize();
    }

    /* --- scaling for inverse ------------------------------------ */
    if (N_inverse != 0) {
        scale_kernel_int<<<blocks, threads>>>(d_data, N, N_inverse, p);
    }

    cudaMemcpy(host_out, d_data, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);  cudaFree(d_data);
    return 0;
}
int ntt_gpu (const uint32_t* host_in, uint32_t* host_out, std::size_t N, std::size_t N_inverse, uint32_t w_primitive_root, uint32_t p)
{ return ntt_gpu_dir(host_in, host_out, N, 0, w_primitive_root, p); }
int intt_gpu(const uint32_t* host_in, uint32_t* host_out, std::size_t N, std::size_t N_inverse, uint32_t w_primitive_root, uint32_t p)
{ return ntt_gpu_dir(host_in, host_out, N, N_inverse, w_primitive_root, p); }

/* ---------- element-wise product (Hadamard) ----------------------- */
__global__ void pointwise_mul_kernel_int(const uint32_t* A,
                                        const uint32_t* B,
                                        uint32_t*       C,
                                        std::size_t     N,
                                        uint32_t        p)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    C[i] = mod_mul(A[i], B[i], p);
}

/* ================================================================== *
   NEW:  radix-2 butterfly for TWO concatenated signals
   data layout: [A0‥A(N-1) | B0‥B(N-1)]
* ================================================================== */
__global__ void ntt_double_stage_kernel(uint32_t*  data,     // 2·N items
    std::size_t N,       // size of ONE poly
    uint32_t    w_root_unity,
    uint32_t    p,
    uint32_t    stage)
{
    /* total butterflies in this stage:   2 · (N/2)  =  N          */
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    /* which polynomial does this thread belong to?  */
    std::size_t halfN      = N >> 1;                // N/2
    std::size_t local_tid  = tid % halfN;           /* butterfly index inside that poly */

    std::size_t mh = 1u << (stage - 1);             /* m/2 */
    std::size_t j  =  local_tid % mh;

    std::size_t idx1 = ((tid >= halfN) ? N : 0) + ((local_tid / mh) * (mh << 1)) + j;
    std::size_t idx2 = idx1 + mh;

    uint32_t u = data[idx1];
    uint32_t t = mod_mul(mod_pow(w_root_unity, j, p), data[idx2], p);

    data[idx1] = mod_add(u, t, p);
    data[idx2] = mod_sub(u, t, p);
}

// ----------------------------------------------------------------------
// bit_reverse_double_kernel: one launch handles both A and B in [0..2N)
// ----------------------------------------------------------------------
__global__ void bit_reverse_double_kernel(const uint32_t* in,
    uint32_t*       out,
    int             lgN,
    std::size_t     N)
{
    unsigned tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = N << 1;
    if (tid >= total) return;

    // which half?
    std::size_t idx    = (tid < N ? tid : tid - N);
    unsigned    rev    = bit_reverse(idx, lgN);
    unsigned    src    = (tid < N ? rev : (N + rev));
    out[tid] = in[src];
}

// ----------------------------------------------------------------------
// scale_double_kernel: scale both A and B by N_inverse in one go
// ----------------------------------------------------------------------
__global__ void scale_double_kernel(uint32_t*     data,
std::size_t   N,
uint32_t      N_inverse,
uint32_t      p)
{
    unsigned tid   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = N << 1;
    if (tid >= total) return;
    data[tid] = mod_mul(data[tid], N_inverse, p);
}


/* ================================================================== *
NEW:  forward / inverse NTT for TWO signals in parallel
* ================================================================== */
static int ntt_double_gpu_dir(const uint32_t* A_host,
    const uint32_t* B_host,
    uint32_t*       A_host_out,
    uint32_t*       B_host_out,
    std::size_t     N,
    std::size_t     N_inverse,
    uint32_t        w_primitive_root,
    uint32_t        p)
{
    if (N == 0 || (N & (N - 1))) return -1;

    const size_t bytes  = N * sizeof(uint32_t);
    const size_t bytes2 = bytes * 2;

    // concat A|B
    uint32_t* d_data{};
    cudaMalloc(&d_data, bytes2);
    cudaMemcpy(d_data,      A_host, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data + N,  B_host, bytes, cudaMemcpyHostToDevice);

    // one bit-reverse for both
    int lgN = static_cast<int>(std::log2(N));
    dim3 thr(256), blkDouble((2 * N + 255) / 256);
    uint32_t* d_tmp{};
    cudaMalloc(&d_tmp, bytes2);

    bit_reverse_double_kernel<<<blkDouble, thr>>>(d_data, d_tmp, lgN, N);
    cudaDeviceSynchronize();
    cudaFree(d_data);
    d_data = d_tmp;

    // iterative NTT stages (unchanged)
    dim3 blkN((N + 255) / 256);
    for (std::size_t m = 2, stage = 1; m <= N; m <<= 1, ++stage) {
    uint32_t w_root_unity = mod_pow(w_primitive_root, (p - 1) / m, p);
    ntt_double_stage_kernel<<<blkN, thr>>>(d_data, N, w_root_unity, p, stage);
    cudaDeviceSynchronize();
    }

    // one scale for both if inverse
    if (N_inverse != 0) {
    scale_double_kernel<<<blkDouble, thr>>>(d_data, N, N_inverse, p);
    cudaDeviceSynchronize();
    }

    // copy-back
    if (A_host_out) cudaMemcpy(A_host_out,     d_data,     bytes, cudaMemcpyDeviceToHost);
    if (B_host_out) cudaMemcpy(B_host_out,     d_data + N, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    return 0;
}


int ntt_double_gpu(const uint32_t* A_host, const uint32_t* B_host,
    uint32_t* A_host_out,    uint32_t* B_host_out,
    std::size_t N,
    uint32_t    w_primitive_root,
    uint32_t    p)
{
    return ntt_double_gpu_dir(A_host, B_host,
    A_host_out, B_host_out,
    N, 0,            /* forward transform -> no scaling */
    w_primitive_root, p);
}

/* ================================================================== *
   helper: build ψⁱ and ψ^{−i}, where ψ^{2N}=1 and ψ^N=−1
   we have g = w_primitive_root is a generator of Z_p^×.
   then ψ = g^{(p−1)/(2N)}, so ψ^2 = g^{(p−1)/N} = ω.
* ================================================================== */
static void precompute_psi_tables(std::size_t            N,
    uint32_t               g,      // your generator
    uint32_t               p,
    std::vector<uint32_t>& psi,
    std::vector<uint32_t>& psi_inv)
{
    // exponent for a 2N-th root: (p−1)/(2N)
    uint32_t exp = static_cast<uint32_t>((p - 1) / (2 * N));
    uint32_t psi_base     = mod_pow(g, exp, p);
    uint32_t psi_base_inv = mod_pow(psi_base, p - 2, p);

    psi.resize(N);
    psi_inv.resize(N);
    psi[0]     = 1u;
    psi_inv[0] = 1u;
    for (std::size_t i = 1; i < N; ++i) {
    psi[i]     = mod_mul(psi[i-1],     psi_base,     p);
    psi_inv[i] = mod_mul(psi_inv[i-1], psi_base_inv, p);
    }
}

/* ================================================================== *
negacyclic polynomial multiplication via two diagonal twists + NTT
C(x) = A(x)·B(x)  mod  (x^N + 1)
* ================================================================== */
int poly_mul_gpu(const uint32_t* A_host,
                       const uint32_t* B_host,
                       uint32_t*       C_host,
                       std::size_t     N,
                       uint32_t        w_primitive_root,
                       uint32_t        w_primitive_root_inv,
                       uint32_t        p)
{
    std::cout << "[DEBUG poly_mul_gpu] N=" << N << " p=" << p << " root=" << w_primitive_root
              << " root_inv=" << w_primitive_root_inv << "\n";

    // 0) build psi tables
    std::vector<uint32_t> psi(N), psi_inv(N);
    {
        uint32_t exp = static_cast<uint32_t>((p - 1) / (2 * N));
        uint32_t psi_base     = mod_pow(w_primitive_root, exp, p);
        uint32_t psi_base_inv = mod_pow(psi_base, p - 2, p);
        psi[0] = psi_inv[0] = 1;
        for (std::size_t i = 1; i < N; ++i) {
            psi[i]     = mod_mul(psi[i - 1], psi_base,     p);
            psi_inv[i] = mod_mul(psi_inv[i - 1], psi_base_inv, p);
        }
    }
    std::cout << "[DEBUG] psi: ";
    for (auto x : psi) std::cout << x << ' ';
    std::cout << "\n[DEBUG] psi_inv: ";
    for (auto x : psi_inv) std::cout << x << ' ';
    std::cout << "\n";

    // 1) pre-twist
    std::vector<uint32_t> A_scaled(N), B_scaled(N);
    for (std::size_t i = 0; i < N; ++i) {
        A_scaled[i] = mod_mul(A_host[i], psi[i], p);
        B_scaled[i] = mod_mul(B_host[i], psi[i], p);
    }
    std::cout << "[DEBUG] A_scaled: ";
    for (auto x : A_scaled) std::cout << x << ' ';
    std::cout << "\n[DEBUG] B_scaled: ";
    for (auto x : B_scaled) std::cout << x << ' ';
    std::cout << "\n";

    // 2) forward NTT on both in parallel
    std::vector<uint32_t> A_ntt(N), B_ntt(N);
    if (ntt_double_gpu(A_scaled.data(), B_scaled.data(), A_ntt.data(), B_ntt.data(), N, w_primitive_root, p)) {
        std::cerr << "[ERROR] forward NTT failed\n";
        return -1;
    }
    std::cout << "[DEBUG] A_ntt: ";
    for (auto x : A_ntt) std::cout << x << ' ';
    std::cout << "\n[DEBUG] B_ntt: ";
    for (auto x : B_ntt) std::cout << x << ' ';
    std::cout << "\n";

    // 3) pointwise product
    const size_t bytes = N * sizeof(uint32_t);
    uint32_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, A_ntt.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_ntt.data(), bytes, cudaMemcpyHostToDevice);

    {
        std::vector<uint32_t> tempA = A_ntt, tempB = B_ntt;
        std::cout << "[DEBUG] pointwise inputs: ";
        for (std::size_t i = 0; i < N; ++i)
            std::cout << tempA[i] << "*" << tempB[i] << ' ';
        std::cout << "\n";
    }

    dim3 thr(256), blk((N + 255) / 256);
    pointwise_mul_kernel_int<<<blk, thr>>>(d_A, d_B, d_C, N, p);
    cudaDeviceSynchronize();

    std::vector<uint32_t> C_pw(N);
    cudaMemcpy(C_pw.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    std::cout << "[DEBUG] C_pointwise: ";
    for (auto x : C_pw) std::cout << x << ' ';
    std::cout << "\n";

    cudaFree(d_A);
    cudaFree(d_B);

    // 4) inverse NTT (cyclic)
    uint32_t N_inv = mod_pow(static_cast<uint32_t>(N), p - 2, p);
    std::vector<uint32_t> C_time(N);
    cudaMemcpy(C_time.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_C);

    std::cout << "[DEBUG] before inverse NTT: ";
    for (auto x : C_time) std::cout << x << ' ';
    std::cout << "\n";

    if (intt_gpu(C_time.data(), C_host, N, N_inv, w_primitive_root_inv, p)) {
        std::cerr << "[ERROR] inverse NTT failed\n";
        return -1;
    }
    std::cout << "[DEBUG] after inverse NTT: ";
    for (std::size_t i = 0; i < N; ++i) std::cout << C_host[i] << ' ';
    std::cout << "\n";

    // 5) post-twist
    for (std::size_t i = 0; i < N; ++i) {
        C_host[i] = mod_mul(C_host[i], psi_inv[i], p);
    }
    std::cout << "[DEBUG] final C: ";
    for (std::size_t i = 0; i < N; ++i) std::cout << C_host[i] << ' ';
    std::cout << "\n";

    return 0;
}

