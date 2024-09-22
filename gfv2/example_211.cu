
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

// Helper function for float to half conversion
__device__ __forceinline__ half float_to_half(float f) {
  return __float2half_rn(f);
}

// Helper function for half to float conversion
__device__ __forceinline__ float half_to_float(half h) {
  return __half2float(h);
}

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const half* a, const half* b, half* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += half_to_float(a[row * k + i]) * half_to_float(b[col * k + i]);
        }
        c[row * n + col] = float_to_half(sum);
    }
}

// CUDA kernel for cross-entropy loss calculation
__global__ void cross_entropy_kernel(const half* out, const int* label, float* loss, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            sum += exp(half_to_float(out[i * batch_size + idx]));
        }
        loss[idx] = -half_to_float(out[idx * batch_size + label[idx]]) + logf(sum);
    }
}

// CUDA kernel for einsum (bnh,bh->bn)
__global__ void einsum_kernel(const half* input, const half* weight, half* output, int batch_size, int n, int h) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < h; ++i) {
            sum += half_to_float(input[row * n * h + col * h + i]) * half_to_float(weight[i * h + col]);
        }
        output[row * n + col] = float_to_half(sum);
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

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract label tensor
    const int* label = va_arg(args, const int*);
    int label_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int n = input_tensor_dim1;
    int h = weight_dim0;
    int output_dim = weight_dim1;

    // Allocate device memory
    half* d_input, *d_weight, *d_out, *d_loss;
    int* d_label;
    cudaMalloc(&d_input, batch_size * n * sizeof(half));
    cudaMalloc(&d_weight, h * output_dim * sizeof(half));
    cudaMalloc(&d_out, batch_size * output_dim * sizeof(half));
    cudaMalloc(&d_loss, batch_size * sizeof(float));
    cudaMalloc(&d_label, batch_size * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, h * output_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_label, label, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // Linear Attention
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_out, batch_size, output_dim, n);

    // Einsum
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_out, d_weight, d_out, batch_size, n, h);

    // Cross-entropy Loss
    cross_entropy_kernel<<<1, 256>>>(d_out, d_label, d_loss, batch_size);

    // Copy result back to host
    cudaMemcpy(output, d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_out);
    cudaFree(d_loss);
    cudaFree(d_label);
}

}  // extern "C"
