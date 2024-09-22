
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for simple matrix transpose
__global__ void simple_matrix_transpose_kernel(const float* matrix, float* output, int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_cols && col < num_rows) {
        output[row * num_rows + col] = matrix[col * num_cols + row];
    }
}

extern "C" {

void simple_matrix_transpose(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract matrix
    const float* matrix = va_arg(args, const float*);
    int matrix_dim0 = va_arg(args, int);
    int matrix_dim1 = va_arg(args, int);

    // Extract output matrix (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int num_rows = matrix_dim0;
    int num_cols = matrix_dim1;

    // Allocate device memory
    float *d_matrix, *d_output;
    cudaMalloc(&d_matrix, num_rows * num_cols * sizeof(float));
    cudaMalloc(&d_output, num_cols * num_rows * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_matrix, matrix, num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (num_cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

    simple_matrix_transpose_kernel<<<numBlocks, threadsPerBlock>>>(
        d_matrix, d_output, num_rows, num_cols
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, num_cols * num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_output);
}

}  // extern "C"
