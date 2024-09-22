
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for matrix transpose
__global__ void transpose_kernel(const float* input_matrix, float* output_matrix, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        output_matrix[col * m + row] = input_matrix[row * n + col];
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

    int m = input_matrix_dim0;
    int n = input_matrix_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, m * n * sizeof(float));
    cudaMalloc(&d_output, n * m * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_matrix, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, m, n
    );

    // Copy result back to host
    cudaMemcpy(output_matrix, d_output, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
