
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for int8 to float conversion and diagflat
__global__ void diagflat_int8_kernel(const int8_t* input_tensor, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i * (n + 1)] = (float)input_tensor[i];
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

    int n = input_tensor_dim0;

    // Allocate device memory
    int8_t *d_input;
    float *d_output;
    cudaMalloc(&d_input, n * sizeof(int8_t));
    cudaMalloc(&d_output, n * n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    diagflat_int8_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, n);

    // Copy result back to host
    cudaMemcpy(output, d_output, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
