
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

// CUDA kernel for 1D convolution with bias and ReLU using bfloat16 (TBC format)
__global__ void conv1d_tbc_kernel_bf16(const float* input, const float* weight, const float* bias, float* output,
                                        int T, int B, int Cin, int Cout, int kernel_size) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int c = threadIdx.z;

    if (t < T && b < B && c < Cout) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            for (int ic = 0; ic < Cin; ic++) {
                int input_idx = (t + k) * B * Cin + b * Cin + ic;
                int weight_idx = c * kernel_size * Cin + k * Cin + ic;
                __nv_bfloat16 a = float_to_bfloat16(input[input_idx]);
                __nv_bfloat16 w = float_to_bfloat16(weight[weight_idx]);
                sum += bfloat16_to_float(__hmul(a, w));
            }
        }
        output[t * B * Cout + b * Cout + c] = fmaxf(bfloat16_to_float(float_to_bfloat16(sum + bias[c])), 0.0f);
    }
}

extern "C" {

void conv_tbc_bf16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int T = va_arg(args, int);
    int B = va_arg(args, int);
    int Cin = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int Cout = va_arg(args, int);
    int kernel_size = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float* d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, T * B * Cin * sizeof(float));
    cudaMalloc(&d_weight, Cout * kernel_size * Cin * sizeof(float));
    cudaMalloc(&d_bias, Cout * sizeof(float));
    cudaMalloc(&d_output, T * B * Cout * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, T * B * Cin * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, Cout * kernel_size * Cin * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, Cout * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 4);
    dim3 numBlocks((T + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (B + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conv1d_tbc_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output, T, B, Cin, Cout, kernel_size
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, T * B * Cout * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
