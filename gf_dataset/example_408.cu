
#include <cuda_runtime.h>
#include <cutlass.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

using namespace cutlass;

// CUDA kernel for feature mixing using Cutlass
__global__ void feature_mixing_kernel(const float* input_tensor, const float* weight1, 
                                      const float* weight2, float* output, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum1 += input_tensor[row * k + i] * weight1[col * k + i];
            sum2 += input_tensor[row * k + i] * weight2[col * k + i];
        }
        output[row * n + col] = fmaxf(sum1 + sum2, 0.0f);  // ReLU activation
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

    // Extract weight1 tensor
    const float* weight1 = va_arg(args, const float*);
    int weight1_dim0 = va_arg(args, int);
    int weight1_dim1 = va_arg(args, int);

    // Extract weight2 tensor
    const float* weight2 = va_arg(args, const float*);
    int weight2_dim0 = va_arg(args, int);
    int weight2_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight1_dim0;

    // Allocate device memory
    float *d_input, *d_weight1, *d_weight2, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight1, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_weight2, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight1, weight1, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight2, weight2, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    feature_mixing_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight1, d_weight2, d_output, batch_size, output_dim, input_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight1);
    cudaFree(d_weight2);
    cudaFree(d_output);
}

}  // extern "C"
