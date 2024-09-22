
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for element-wise addition using bfloat16
__global__ void elementwise_addition_kernel_bf16(const float* input_tensor1, const float* input_tensor2, float* output, 
                                            int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        __nv_bfloat16 a = float_to_bfloat16(input_tensor1[row * n + col]);
        __nv_bfloat16 b = float_to_bfloat16(input_tensor2[row * n + col]);
        sum += bfloat16_to_float(__hadd(a, b));
        output[row * n + col] = sum;
    }
}

extern "C" {

void elementwise_addition(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor1 = va_arg(args, const float*);
    int input_tensor1_dim0 = va_arg(args, int);
    int input_tensor1_dim1 = va_arg(args, int);

    const float* input_tensor2 = va_arg(args, const float*);
    int input_tensor2_dim0 = va_arg(args, int);
    int input_tensor2_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor1_dim0;
    int input_dim = input_tensor1_dim1;

    // Check if input tensors have the same shape
    if (input_tensor1_dim0 != input_tensor2_dim0 || input_tensor1_dim1 != input_tensor2_dim1) {
        printf("Error: Input tensors must have the same shape\n");
        return;
    }

    // Allocate device memory
    float *d_input_tensor1, *d_input_tensor2, *d_output;
    cudaMalloc(&d_input_tensor1, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_input_tensor2, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input_tensor1, input_tensor1, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_tensor2, input_tensor2, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    elementwise_addition_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input_tensor1, d_input_tensor2, d_output, batch_size, input_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input_tensor1);
    cudaFree(d_input_tensor2);
    cudaFree(d_output);
}

}  // extern "C"