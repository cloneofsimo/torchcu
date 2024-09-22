
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for simple matrix subtraction
__global__ void matrix_subtraction_kernel(const float* matrix1, const float* matrix2, float* output, 
                                          int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        output[row * n + col] = matrix1[row * n + col] - matrix2[row * n + col];
    }
}

extern "C" {

void simple_matrix_subtraction(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract first matrix
    const float* matrix1 = va_arg(args, const float*);
    int matrix1_dim0 = va_arg(args, int);
    int matrix1_dim1 = va_arg(args, int);

    // Extract second matrix
    const float* matrix2 = va_arg(args, const float*);
    int matrix2_dim0 = va_arg(args, int);
    int matrix2_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int m = matrix1_dim0;
    int n = matrix1_dim1;

    // Allocate device memory
    float *d_matrix1, *d_matrix2, *d_output;
    cudaMalloc(&d_matrix1, m * n * sizeof(float));
    cudaMalloc(&d_matrix2, m * n * sizeof(float));
    cudaMalloc(&d_output, m * n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_matrix1, matrix1, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_subtraction_kernel<<<numBlocks, threadsPerBlock>>>(
        d_matrix1, d_matrix2, d_output, m, n
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_output);
}

}  // extern "C"
