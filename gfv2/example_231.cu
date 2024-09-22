
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for chain matrix multiplication using bfloat16
__global__ void chain_matmul_kernel_bf16(const float* input1, const float* input2, const float* input3,
                                        float* output, int m, int k1, int k2, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k1; ++i) {
            for (int j = 0; j < k2; ++j) {
                __nv_bfloat16 a = float_to_bfloat16(input1[row * k1 + i]);
                __nv_bfloat16 b = float_to_bfloat16(input2[i * k2 + j]);
                __nv_bfloat16 c = float_to_bfloat16(input3[j * n + col]);
                sum += bfloat16_to_float(__hmul(__hmul(a, b), c));
            }
        }
        output[row * n + col] = sum;
    }
}

extern "C" {

void chain_matmul_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input1 = va_arg(args, const float*);
    int input1_dim0 = va_arg(args, int);
    int input1_dim1 = va_arg(args, int);

    const float* input2 = va_arg(args, const float*);
    int input2_dim0 = va_arg(args, int);
    int input2_dim1 = va_arg(args, int);

    const float* input3 = va_arg(args, const float*);
    int input3_dim0 = va_arg(args, int);
    int input3_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int m = input1_dim0;
    int k1 = input1_dim1;
    int k2 = input2_dim1;
    int n = input3_dim1;

    // Allocate device memory
    float *d_input1, *d_input2, *d_input3, *d_output;
    cudaMalloc(&d_input1, m * k1 * sizeof(float));
    cudaMalloc(&d_input2, k1 * k2 * sizeof(float));
    cudaMalloc(&d_input3, k2 * n * sizeof(float));
    cudaMalloc(&d_output, m * n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input1, input1, m * k1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2, k1 * k2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input3, input3, k2 * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    chain_matmul_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input1, d_input2, d_input3, d_output, m, k1, k2, n
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_input3);
    cudaFree(d_output);
}

}  // extern "C"
