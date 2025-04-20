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
__global__ void poly_sum_kernel(int *input1, int *input2, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (; tid < size; tid += stride) {
        input2[tid] = input1[tid] + input2[tid];
    }
}

extern "C" int get_max_threads_per_block() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, /*device=*/0);
    return prop.maxThreadsPerBlock;
}

// Wrapper function to call the CUDA kernel
extern "C" void poly_sum(int *input1, int *input2, int size) {
    // Allocate device memory
    int *d_input1, *d_input2;
    CUDA_CHECK(cudaMalloc((void**)&d_input1, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_input2, size * sizeof(int)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input1, input1, size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input2, input2, size * sizeof(int), cudaMemcpyHostToDevice));

    // Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Query max active blocks per multiprocessor
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, poly_sum_kernel));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << prop.maxGridSize[0];
    // In case number of needed blocks exceeds hardware limit
    blocksPerGrid = std::min(blocksPerGrid, prop.maxGridSize[0]);
    poly_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, size);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(input2, d_input2, size * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
}
