
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cudnn.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

extern "C" {

__global__ void diagflat_einsum_kernel(const float* input, const float* weight, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * n) {
        int row = i / n;
        int col = i % n;
        if (row == col) {
            output[i] = input[row] * weight[col * n + row];
        }
    }
}

// CUDA kernel for calculating gradients of diagflat and weight_t
__global__ void diagflat_einsum_backward_kernel(const float* grad_output, const float* weight, float* grad_diag, float* grad_weight, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * n) {
        int row = i / n;
        int col = i % n;
        grad_diag[row] += grad_output[row * n + col] * weight[col * n + row];
        grad_weight[col * n + row] += grad_output[row * n + col] * input[row];
    }
}

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input = va_arg(args, const float*);
    int input_dim = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    float* output = va_arg(args, float*);

    float* grad_diag = va_arg(args, float*);
    float* grad_weight = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_weight, *d_output, *d_grad_diag, *d_grad_weight;
    cudaMalloc(&d_input, input_dim * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * sizeof(float));
    cudaMalloc(&d_output, input_dim * weight_dim1 * sizeof(float));
    cudaMalloc(&d_grad_diag, input_dim * sizeof(float));
    cudaMalloc(&d_grad_weight, weight_dim0 * weight_dim1 * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input, input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel for diagflat_einsum
    dim3 threadsPerBlock(256);
    dim3 numBlocks((input_dim * weight_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x);
    diagflat_einsum_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_output, input_dim);

    // Copy output to host
    cudaMemcpy(output, d_output, input_dim * weight_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Launch kernel for backward pass
    diagflat_einsum_backward_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_weight, d_grad_diag, d_grad_weight, input_dim);

    // Copy gradients to host
    cudaMemcpy(grad_diag, d_grad_diag, input_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_weight, d_grad_weight, weight_dim0 * weight_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaFree(d_grad_diag);
    cudaFree(d_grad_weight);
}

}
