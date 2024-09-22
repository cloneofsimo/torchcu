
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for calculating the mean of a tensor using fp16
__global__ void mean_kernel_fp16(const float* input_tensor, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // Load input as half precision
        __half input_half = __float2half_rn(input_tensor[i]);
        // Accumulate sum using half precision
        __half sum = __int2half_rn(0);
        sum = __hadd(sum, input_half);
        // Store result as float
        output[0] = __half2float(sum) / size;
    }
}

extern "C" {

void mean_fp16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int size = input_tensor_dim0;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    mean_kernel_fp16<<<numBlocks, threadsPerBlock>>>(d_input, d_output, size);

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
