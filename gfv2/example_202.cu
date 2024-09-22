
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for fused linear (matrix multiplication + bias) and ReLU using int8
__global__ void fused_linear_relu_kernel_int8(const int8_t* input_tensor, const int8_t* weight, const int8_t* bias, 
                                        float* output, int m, int n, int k, int bias_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += input_tensor[row * k + i] * weight[col * k + i];
        }
        output[row * n + col] = fmaxf(half_to_float(sum + bias[col]), 0.0f);
    }
}

extern "C" {

void fused_linear_int8_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const int8_t* input_tensor = va_arg(args, const int8_t*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weight tensor
    const int8_t* weight = va_arg(args, const int8_t*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract bias tensor
    const int8_t* bias = va_arg(args, const int8_t*);
    int bias_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;

    // Allocate device memory
    int8_t *d_input, *d_weight, *d_bias;
    float *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(int8_t));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(int8_t));
    cudaMalloc(&d_bias, bias_size * sizeof(int8_t));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_size * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    fused_linear_relu_kernel_int8<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output, batch_size, output_dim, input_dim, bias_size
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
