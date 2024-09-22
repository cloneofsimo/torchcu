
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for matrix transpose
__global__ void transpose_kernel(const float* input, float* output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        output[col * m + row] = input[row * n + col];
    }
}

extern "C" {

void simple_matrix_transpose(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int num_rows = input_dim0;
    int num_cols = input_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, num_rows * num_cols * sizeof(float));
    cudaMalloc(&d_output, num_cols * num_rows * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, num_rows, num_cols
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, num_cols * num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
