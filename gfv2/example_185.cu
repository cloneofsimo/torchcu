
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for KL divergence between normal distributions
__global__ void kl_div_kernel_fp16(const float* input_tensor1, const float* input_tensor2, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        half mean1 = float_to_half(input_tensor1[i]);
        half mean2 = float_to_half(input_tensor2[i]);
        half log_var1 = float_to_half(0.0f); // Assuming variance is 1
        half log_var2 = float_to_half(0.0f); // Assuming variance is 1

        half kl_div = 0.5f * ((log_var2 - log_var1) + (exp(log_var1) + (mean1 - mean2) * (mean1 - mean2)) / exp(log_var2) - 1.0f);
        output[0] += half_to_float(kl_div);
    }
}

extern "C" {

void kl_div_fp16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor1 = va_arg(args, const float*);
    int input_tensor1_dim0 = va_arg(args, int);

    // Extract second input tensor
    const float* input_tensor2 = va_arg(args, const float*);
    int input_tensor2_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int size = input_tensor1_dim0;

    // Allocate device memory
    float *d_input1, *d_input2, *d_output;
    cudaMalloc(&d_input1, size * sizeof(float));
    cudaMalloc(&d_input2, size * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input1, input_tensor1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input_tensor2, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    kl_div_kernel_fp16<<<numBlocks, threadsPerBlock>>>(d_input1, d_input2, d_output, size);

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}

}  // extern "C"
