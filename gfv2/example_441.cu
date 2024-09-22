
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for baddbmm and masked selection
__global__ void baddbmm_masked_kernel(const float* input_tensor, const float* weight, const bool* mask, float* output, 
                                        int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = input_tensor[row * n + col]; // Initialize with input value
        for (int i = 0; i < k; ++i) {
            sum += input_tensor[row * k + i] * weight[col * k + i];
        }
        output[row * n + col] = sum;

        // Apply mask
        if (!mask[row * n + col]) {
            output[row * n + col] = 0.0f;
        }
    }
}

// CUDA kernel for reduction (sum)
__global__ void reduction_kernel(const float* output, float* result, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m * n) {
        atomicAdd(result, output[idx]);
    }
}

extern "C" {

void complex_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract mask tensor
    const bool* mask = va_arg(args, const bool*);
    int mask_dim0 = va_arg(args, int);
    int mask_dim1 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    bool *d_mask;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));
    cudaMalloc(&d_mask, batch_size * input_dim * sizeof(bool));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, batch_size * input_dim * sizeof(bool), cudaMemcpyHostToDevice);

    // Launch baddbmm and masked selection kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    baddbmm_masked_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_mask, d_output, batch_size, output_dim, input_dim
    );

    // Allocate device memory for result
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));

    // Launch reduction kernel
    numBlocks = (batch_size * output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x;

    reduction_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_result, batch_size, output_dim);

    // Copy result back to host
    cudaMemcpy(output, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaFree(d_mask);
    cudaFree(d_result);
}

}  // extern "C"
