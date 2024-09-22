
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

__global__ void pairwise_distance_bf16_kernel(const float* input1, const float* input2, float* output, 
                                            int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input1[row * k + i]);
            __nv_bfloat16 b = float_to_bfloat16(input2[col * k + i]);
            sum += bfloat16_to_float(__hmul(a - b, a - b)); 
        }
        output[row * n + col] = sum;
    }
}

extern "C" {

void pairwise_distance_bf16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input1 = va_arg(args, const float*);
    int input1_dim0 = va_arg(args, int);
    int input1_dim1 = va_arg(args, int);

    const float* input2 = va_arg(args, const float*);
    int input2_dim0 = va_arg(args, int);
    int input2_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int m = input1_dim0;
    int n = input2_dim0;
    int k = input1_dim1;

    // Allocate device memory
    float *d_input1, *d_input2, *d_output;
    cudaMalloc(&d_input1, m * k * sizeof(float));
    cudaMalloc(&d_input2, n * k * sizeof(float));
    cudaMalloc(&d_output, m * n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input1, input1, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2, n * k * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    pairwise_distance_bf16_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input1, d_input2, d_output, m, n, k
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}

}  // extern "C"
