
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For half-precision (FP16) support
#include <device_launch_parameters.h>
#include <stdarg.h>  // For va_list, va_start, va_end

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);  // Round-to-nearest conversion
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for batched matrix multiplication and addition with bias using FP16
__global__ void baddbmm_kernel_fp16(const float* input_tensor, const float* weight, const float* bias,
                                        float* output, int batch_size, int m, int k, int n) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = threadIdx.z;

    if (batch_idx < batch_size && row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            half a = float_to_half(input_tensor[batch_idx * m * k + row * k + i]);
            half b = float_to_half(weight[batch_idx * k * n + i * n + col]);
            sum += half_to_float(__hmul(a, b));
        }
        output[batch_idx * m * n + row * n + col] = sum + bias[batch_idx * n + col];
    }
}

extern "C" {

void torch_baddbmm_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);
    int bias_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int m = input_tensor_dim1;
    int k = input_tensor_dim2;
    int n = weight_dim2;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * m * k * sizeof(float));
    cudaMalloc(&d_weight, batch_size * k * n * sizeof(float));
    cudaMalloc(&d_bias, batch_size * n * sizeof(float));
    cudaMalloc(&d_output, batch_size * m * n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, batch_size * k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, batch_size * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (n + threadsPerBlock.z - 1) / threadsPerBlock.z);

    baddbmm_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output, batch_size, m, k, n
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
