
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for matrix transpose using bfloat16
__global__ void transpose_kernel_bf16(const float* input_matrix, float* output_matrix, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        output_matrix[col * m + row] = input_matrix[row * n + col];
    }
}

// CUDA kernel for matrix transpose using bfloat16
__global__ void transpose_kernel_bf16_2(const float* input_matrix, float* output_matrix, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        output_matrix[row * n + col] = input_matrix[col * m + row];
    }
}

extern "C" {

void matrix_transpose(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_matrix = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output_matrix = va_arg(args, float*);

    va_end(args);

    int batch_size = input_dim0;
    int input_dim = input_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_matrix, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, input_dim
    );

    // Copy result back to host
    cudaMemcpy(output_matrix, d_output, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

void matrix_transpose_2(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_matrix = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output_matrix = va_arg(args, float*);

    va_end(args);

    int batch_size = input_dim0;
    int input_dim = input_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_matrix, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose_kernel_bf16_2<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, input_dim
    );

    // Copy result back to host
    cudaMemcpy(output_matrix, d_output, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
