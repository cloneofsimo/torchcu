
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert int8_t to float
__device__ __forceinline__ float int8_to_float(int8_t val) {
    return static_cast<float>(val);
}

__global__ void mean_int8_kernel(const int8_t* input, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        atomicAdd(output, int8_to_float(input[i]));
    }
}

extern "C" {

void mean_int8_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const int8_t* input = va_arg(args, const int8_t*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int size = input_dim0 * input_dim1;

    // Allocate device memory
    int8_t *d_input;
    float *d_output;
    cudaMalloc(&d_input, size * sizeof(int8_t));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, size * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    mean_int8_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, size);

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate the final mean
    *output /= static_cast<float>(size);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
