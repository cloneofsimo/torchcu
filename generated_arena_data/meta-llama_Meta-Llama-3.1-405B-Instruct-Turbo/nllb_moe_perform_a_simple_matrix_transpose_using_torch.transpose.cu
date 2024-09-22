
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  

// CUDA kernel for matrix transpose
__global__ void matrix_transpose_kernel(const float* input_matrix, float* output_matrix, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output_matrix[col * rows + row] = input_matrix[row * cols + col];
    }
}

extern "C" {

void simple_matrix_transpose(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input matrix
    const float* input_matrix = va_arg(args, const float*);
    int input_matrix_dim0 = va_arg(args, int);
    int input_matrix_dim1 = va_arg(args, int);

    // Extract output matrix (assuming it's preallocated)
    float* output_matrix = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_matrix_dim0 * input_matrix_dim1 * sizeof(float));
    cudaMalloc(&d_output, input_matrix_dim0 * input_matrix_dim1 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_matrix, input_matrix_dim0 * input_matrix_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_matrix_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_matrix_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, input_matrix_dim0, input_matrix_dim1
    );

    // Copy result back to host
    cudaMemcpy(output_matrix, d_output, input_matrix_dim0 * input_matrix_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
