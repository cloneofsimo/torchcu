
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for matrix addition
__global__ void matrix_addition_kernel(const float* matrix1, const float* matrix2, float* output, int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows && col < num_cols) {
        int index = row * num_cols + col;
        output[index] = matrix1[index] + matrix2[index];
    }
}

extern "C" {

void simple_matrix_addition(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract matrix1
    const float* matrix1 = va_arg(args, const float*);
    int matrix1_dim0 = va_arg(args, int);
    int matrix1_dim1 = va_arg(args, int);

    // Extract matrix2
    const float* matrix2 = va_arg(args, const float*);
    int matrix2_dim0 = va_arg(args, int);
    int matrix2_dim1 = va_arg(args, int);

    // Extract output matrix (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int num_rows = matrix1_dim0;
    int num_cols = matrix1_dim1;

    // Allocate device memory
    float *d_matrix1, *d_matrix2, *d_output;
    cudaMalloc(&d_matrix1, num_rows * num_cols * sizeof(float));
    cudaMalloc(&d_matrix2, num_rows * num_cols * sizeof(float));
    cudaMalloc(&d_output, num_rows * num_cols * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_matrix1, matrix1, num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_addition_kernel<<<numBlocks, threadsPerBlock>>>(
        d_matrix1, d_matrix2, d_output, num_rows, num_cols
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, num_rows * num_cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_output);
}

}  // extern "C"
