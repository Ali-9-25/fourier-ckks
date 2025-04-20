#include <cstdio>
#include <iostream>
#define CUDA_CHECK(call)                                                     \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(err));                   \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// CUDA kernel function
__global__ void polynomial_mod_kernel(int *polynomial, int size, int coeff_mod) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (; tid < size; tid += stride) {
        polynomial[tid] %= coeff_mod;
    }
}

// Wrapper function to call the CUDA kernel
extern "C" void polynomial_mod(int *polynomial, int size, int coeff_mod) {
    // Allocate device memory
    int *d_polynomial = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_polynomial, size * sizeof(int)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_polynomial, polynomial, size * sizeof(int), cudaMemcpyHostToDevice));

    // Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Query max active blocks per multiprocessor
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, polynomial_mod_kernel));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << prop.maxGridSize[0];
    // In case number of needed blocks exceeds hardware limit
    blocksPerGrid = std::min(blocksPerGrid, prop.maxGridSize[0]);
    
    polynomial_mod_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_polynomial, size, coeff_mod);

    CUDA_CHECK(cudaDeviceSynchronize());
    //CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(polynomial, d_polynomial, size * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_polynomial));
}
