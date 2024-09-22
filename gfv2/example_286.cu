
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for diagflat operation
__global__ void diagflat_kernel_int8(const int8_t* input_tensor, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i * (size + 1)] = (float)input_tensor[i];
    }
}

extern "C" {

void diagflat_int8_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int size = input_tensor_dim0;

    // Allocate device memory
    int8_t *d_input;
    float *d_output;
    cudaMalloc(&d_input, size * sizeof(int8_t));
    cudaMalloc(&d_output, size * size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    diagflat_kernel_int8<<<numBlocks, threadsPerBlock>>>(d_input, d_output, size);

    // Copy result back to host
    cudaMemcpy(output, d_output, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

} // extern "C"
