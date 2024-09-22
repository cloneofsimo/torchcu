
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for matrix multiplication using float32
__global__ void matmul_kernel(const float* matrix1, const float* matrix2, float* output, 
                              int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += matrix1[row * k + i] * matrix2[i * n + col];
        }
        output[row * n + col] = sum;
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

    int batch_size = matrix1_dim0;
    int input_dim = matrix1_dim1;
    int output_dim = matrix2_dim1;

    // Allocate device memory
    float *d_matrix1, *d_matrix2, *d_output;
    cudaMalloc(&d_matrix1, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_matrix2, input_dim * output_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_matrix1, matrix1, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, input_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        d_matrix1, d_matrix2, d_output, batch_size, output_dim, input_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_output);
}

}  // extern "C"
