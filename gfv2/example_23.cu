
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// CUDA kernel for outer product using FP16
__global__ void outer_product_kernel_fp16(const float* input_tensor1, const float* input_tensor2, 
                                        half* output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        output[row * n + col] = float_to_half(input_tensor1[row] * input_tensor2[col]);
    }
}

extern "C" {

void outer_product_fp16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor 1
    const float* input_tensor1 = va_arg(args, const float*);
    int input_tensor1_dim = va_arg(args, int);

    // Extract input tensor 2
    const float* input_tensor2 = va_arg(args, const float*);
    int input_tensor2_dim = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    half* output = va_arg(args, half*);

    va_end(args);

    // Allocate device memory
    float *d_input1, *d_input2;
    half *d_output;
    cudaMalloc(&d_input1, input_tensor1_dim * sizeof(float));
    cudaMalloc(&d_input2, input_tensor2_dim * sizeof(float));
    cudaMalloc(&d_output, input_tensor1_dim * input_tensor2_dim * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input1, input_tensor1, input_tensor1_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input_tensor2, input_tensor2_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_tensor2_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_tensor1_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);

    outer_product_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_input1, d_input2, d_output, input_tensor1_dim, input_tensor2_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor1_dim * input_tensor2_dim * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}

}  // extern "C"
