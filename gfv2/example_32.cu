
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

#include "cutlass.h"
#include "cutlass/cutlass.h"

#include "cutlass/conv/kernel.h"
#include "cutlass/conv/device/kernel.h"
#include "cutlass/conv/device/gemm.h"
#include "cutlass/conv/device/gemm_sm75.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_sm75.h"
#include "cutlass/conv/gemm/gemm_plan.h"
#include "cutlass/conv/gemm/gemm_plan_sm75.h"

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

template <typename T>
__global__ void unique_and_sum_kernel(const T* input_tensor, const T* mask, T* output, 
                                        int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        T unique_val = input_tensor[idx];
        T sum = 0;
        for (int j = 0; j < M; ++j) {
            if (input_tensor[j] == unique_val) {
                sum += mask[j];
            }
        }
        output[idx] = sum;
    }
}

extern "C" {

void unique_and_sum(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract mask tensor
    const float* mask = va_arg(args, const float*);
    int mask_dim0 = va_arg(args, int);
    int mask_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int N = input_tensor_dim0 * input_tensor_dim1;
    int M = mask_dim0 * mask_dim1;

    // Allocate device memory
    float *d_input, *d_mask, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_mask, M * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, M * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    unique_and_sum_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_mask, d_output, N, M
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
}

}  // extern "C"
