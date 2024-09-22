
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for matrix transpose
__global__ void matrix_transpose_kernel(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

extern "C" {

void simple_matrix_transpose(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input matrix
    const float* matrix = va_arg(args, const float*);
    int matrix_dim0 = va_arg(args, int);
    int matrix_dim1 = va_arg(args, int);

    // Extract output matrix (assuming it's preallocated)
    float* output_matrix = va_arg(args, float*);

    va_end(args);

    int width = matrix_dim1;
    int height = matrix_dim0;

    // Allocate device memory
    float *d_matrix, *d_output_matrix;
    cudaMalloc(&d_matrix, width * height * sizeof(float));
    cudaMalloc(&d_output_matrix, width * height * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_matrix, matrix, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_output_matrix, width, height);

    // Copy result back to host
    cudaMemcpy(output_matrix, d_output_matrix, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_output_matrix);
}

}  // extern "C"
