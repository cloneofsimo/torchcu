
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for matrix transpose
__global__ void matrix_transpose_kernel(const float* input_matrix, float* output_matrix, int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {
        output_matrix[col * num_rows + row] = input_matrix[row * num_cols + col];
    }
}

extern "C" {

void simple_matrix_transpose(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_matrix = va_arg(args, const float*);
    int input_matrix_dim0 = va_arg(args, int);
    int input_matrix_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output_matrix = va_arg(args, float*);

    va_end(args);

    int num_rows = input_matrix_dim0;
    int num_cols = input_matrix_dim1;

    // Allocate device memory
    float *d_input_matrix, *d_output_matrix;
    cudaMalloc(&d_input_matrix, num_rows * num_cols * sizeof(float));
    cudaMalloc(&d_output_matrix, num_rows * num_cols * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input_matrix, input_matrix, num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input_matrix, d_output_matrix, num_rows, num_cols
    );

    // Copy result back to host
    cudaMemcpy(output_matrix, d_output_matrix, num_rows * num_cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input_matrix);
    cudaFree(d_output_matrix);
}

}  // extern "C"
