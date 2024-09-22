
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <stdarg.h>

#define SELU_ALPHA 1.6732632423543772848170429916717
#define SELU_SCALE 1.0507009873554804934193349852946

__device__ float selu(float x) {
    return (x > 0.0f) ? SELU_SCALE * x : SELU_SCALE * SELU_ALPHA * (expf(x) - 1.0f);
}

__global__ void my_function_kernel(const float* input, const float* weight, const float* bias, float* output, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += input[row * k + i] * weight[col * k + i];
        }
        sum += bias[col];
        output[row * n + col] = selu(sum);
    }
}

__global__ void adaptive_max_pool_kernel(const float* input, float* output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float max_val = input[row * n + col];
        for (int i = 1; i < n; ++i) {
            max_val = fmaxf(max_val, input[row * n + col + i]);
        }
        output[row] = max_val;
    }
}

extern "C" {
    void my_function(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract input tensor
        const float* input = va_arg(args, const float*);
        int input_dim0 = va_arg(args, int);
        int input_dim1 = va_arg(args, int);

        // Extract weight tensor
        const float* weight = va_arg(args, const float*);
        int weight_dim0 = va_arg(args, int);
        int weight_dim1 = va_arg(args, int);

        // Extract bias tensor
        const float* bias = va_arg(args, const float*);
        int bias_dim = va_arg(args, int);

        // Extract output tensor (assuming it's preallocated)
        float* output = va_arg(args, float*);

        va_end(args);

        int m = input_dim0;
        int k = input_dim1;
        int n = weight_dim0;

        // Allocate device memory
        float *d_input, *d_weight, *d_bias, *d_output, *d_output_pool;
        cudaMalloc(&d_input, m * k * sizeof(float));
        cudaMalloc(&d_weight, n * k * sizeof(float));
        cudaMalloc(&d_bias, n * sizeof(float));
        cudaMalloc(&d_output, m * n * sizeof(float));
        cudaMalloc(&d_output_pool, m * sizeof(float));

        // Copy input data to device
        cudaMemcpy(d_input, input, m * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, weight, n * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, bias, n * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel for matrix multiplication and SELU activation
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
        my_function_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_bias, d_output, m, n, k);

        // Launch kernel for adaptive max pooling
        numBlocks = (m + threadsPerBlock.x - 1) / threadsPerBlock.x;
        adaptive_max_pool_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_output_pool, m, n);

        // Perform inplace division on the pooled output
        cudaMemcpy(d_output, d_output_pool, m * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(output, d_output, m * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_weight);
        cudaFree(d_bias);
        cudaFree(d_output);
        cudaFree(d_output_pool);
    }
}
