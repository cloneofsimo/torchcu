
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for pruned linear transformation and int8 quantization
__global__ void pruned_linear_int8_kernel(const float* input_tensor, const float* weight, 
                                         const bool* pruning_mask, int8_t* output,
                                         int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            if (pruning_mask[col * k + i]) { // Apply pruning mask
                sum += input_tensor[row * k + i] * weight[col * k + i];
            }
        }
        output[row * n + col] = (int8_t)sum; // Quantize to int8
    }
}

extern "C" {

void pruned_linear_int8(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract pruning mask
    const bool* pruning_mask = va_arg(args, const bool*);

    // Extract output tensor (assuming it's preallocated)
    int8_t* output = va_arg(args, int8_t*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;

    // Allocate device memory
    float *d_input, *d_weight;
    bool *d_pruning_mask;
    int8_t *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_pruning_mask, output_dim * input_dim * sizeof(bool));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(int8_t));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pruning_mask, pruning_mask, output_dim * input_dim * sizeof(bool), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32); // Adjust block size for optimal performance
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    pruned_linear_int8_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_pruning_mask, d_output, batch_size, output_dim, input_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_pruning_mask);
    cudaFree(d_output);
}

}  // extern "C"
