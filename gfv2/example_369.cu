
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// Helper function for weight standardization
__device__ __forceinline__ float standardize_weight(float weight, float mean, float std) {
    return (weight - mean) / std;
}

// CUDA kernel for linear transformation with weight standardization and ReLU activation
__global__ void linear_relu_kernel(const float* input_tensor, const float* weights, const float* bias, float* output,
                                   int m, int n, int k, const float* weight_means, const float* weight_stds) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += input_tensor[row * k + i] * standardize_weight(weights[col * k + i], weight_means[i], weight_stds[i]);
        }
        output[row * n + col] = fmaxf(sum + bias[col], 0.0f);
    }
}

extern "C" {

void my_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weights tensor
    const float* weights = va_arg(args, const float*);
    int weights_dim0 = va_arg(args, int);
    int weights_dim1 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weights_dim0;

    // Allocate device memory
    float *d_input, *d_weights, *d_bias, *d_output, *d_weight_means, *d_weight_stds;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weights, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_bias, output_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));
    cudaMalloc(&d_weight_means, input_dim * sizeof(float));
    cudaMalloc(&d_weight_stds, input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate weight means and standard deviations on the device
    cudaMemcpy(d_weight_means, weight_means, input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_stds, weight_stds, input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    linear_relu_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weights, d_bias, d_output, batch_size, output_dim, input_dim, d_weight_means, d_weight_stds
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaFree(d_weight_means);
    cudaFree(d_weight_stds);
}

}  // extern "C"
