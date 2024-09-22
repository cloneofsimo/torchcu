
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* matrix1, const float* matrix2, float* output, 
                              int num_rows_matrix1, int num_cols_matrix1, int num_cols_matrix2) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows_matrix1 && col < num_cols_matrix2) {
        float sum = 0.0f;
        for (int k = 0; k < num_cols_matrix1; ++k) {
            sum += matrix1[row * num_cols_matrix1 + k] * matrix2[k * num_cols_matrix2 + col];
        }
        output[row * num_cols_matrix2 + col] = sum;
    }
}

extern "C" {

void simple_matrix_multiplication(int num_args, ...) {
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

    int num_rows_matrix1 = matrix1_dim0;
    int num_cols_matrix1 = matrix1_dim1;
    int num_cols_matrix2 = matrix2_dim1;

    // Allocate device memory
    float *d_matrix1, *d_matrix2, *d_output;
    cudaMalloc(&d_matrix1, num_rows_matrix1 * num_cols_matrix1 * sizeof(float));
    cudaMalloc(&d_matrix2, matrix2_dim0 * matrix2_dim1 * sizeof(float));
    cudaMalloc(&d_output, num_rows_matrix1 * num_cols_matrix2 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_matrix1, matrix1, num_rows_matrix1 * num_cols_matrix1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, matrix2_dim0 * matrix2_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_cols_matrix2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_rows_matrix1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        d_matrix1, d_matrix2, d_output, num_rows_matrix1, num_cols_matrix1, num_cols_matrix2
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, num_rows_matrix1 * num_cols_matrix2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_output);
}

}  // extern "C"
