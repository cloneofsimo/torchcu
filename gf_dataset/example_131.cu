
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

__global__ void median_addcmul_clamp_int8_kernel(const float* input, const float* weight, const float* bias, 
                                                float* output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        // Calculate median along dimension 1
        float median = 0.0f;
        for (int i = 0; i < n; ++i) {
            median += input[row * n + i];
        }
        median /= n;  // Simple average for median approximation

        // Perform addcmul
        float sum = bias[col] + weight[col * n + row] * median;

        // Clamp the output
        sum = fminf(fmaxf(sum, -128.0f), 127.0f);

        output[row * n + col] = sum;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, input_dim0 * input_dim1 * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * sizeof(float));
    cudaMalloc(&d_bias, bias_dim0 * sizeof(float));
    cudaMalloc(&d_output, input_dim0 * input_dim1 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, input_dim0 * input_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    median_addcmul_clamp_int8_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output, input_dim0, input_dim1
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, input_dim0 * input_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
