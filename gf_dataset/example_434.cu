
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv_tensor_op.h"
#include "cutlass/conv/kernel/default_conv_tensor_op_fwd.h"

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for softmax, std, and hardtanh operations
__global__ void fused_softmax_std_hardtanh_kernel_bf16(const float* input_tensor, float* output, 
                                                      int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < 1) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input_tensor[row * n + i]);
            sum += bfloat16_to_float(exp(a));
        }

        float softmax_output = 0.0f;
        float variance = 0.0f;
        for (int i = 0; i < n; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input_tensor[row * n + i]);
            softmax_output = exp(a) / sum;
            variance += pow(softmax_output, 2);
        }

        output[row * 1 + col] = fmaxf(fminf(sqrt(variance), 2.0f), -2.0f);
    }
}


extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * 1 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    fused_softmax_std_hardtanh_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, input_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * 1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
