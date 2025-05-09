// multi_ntt_wrapper.cu
#include "multi_ntt.cu"   // your NTT implementation

extern "C" {

// A simple C‐callable wrapper around poly_mul_multi_gpu
__declspec(dllexport)
int poly_mul_multi_c(const uint32_t* A_host,
                     const uint32_t* B_host,
                     uint32_t*       C_host,
                     uint32_t        N,
                     const uint32_t* primitive_roots,
                     const uint32_t* primitive_roots_inv,
                     const uint32_t* primes,
                     uint32_t        num_polys)
{
    // cast away const because the core API is not const‐correct
    return poly_mul_multi_gpu(
        A_host,
        B_host,
        C_host,
        N,
        const_cast<uint32_t*>(primitive_roots),
        const_cast<uint32_t*>(primitive_roots_inv),
        const_cast<uint32_t*>(primes),
        num_polys
    );
}

} // extern "C"
