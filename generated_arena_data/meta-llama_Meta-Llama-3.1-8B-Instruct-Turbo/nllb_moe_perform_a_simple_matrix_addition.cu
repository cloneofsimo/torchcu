
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for matrix addition
__global__ void matadd_kernel(const float* matrix1, const float* matrix2, float* output, 
                           int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows && col < num_cols) {
        output[row * num_cols + col] = matrix1[row * num_cols + col] + matrix2[row * num_cols + col];
    }
}

extern "C" {

void simple_matrix_addition(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* matrix1 = va_arg(args, const float*);
    int matrix1_dim0 = va_arg(args, int);
    int matrix1_dim1 = va_arg(args, int);

    const float* matrix2 = va_arg(args, const float*);
    int matrix2_dim0 = va_arg(args, int);
    int matrix2_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Check if the matrices have the same dimensions
    if (matrix2_dim0 != matrix1_dim0 || matrix2_dim1 != matrix1_dim1) {
        printf("Error: The matrices must have the same dimensions.\n");
        return;
    }

    // Allocate device memory
    float *d_matrix1, *d_matrix2, *d_output;
    cudaMalloc(&d_matrix1, matrix1_dim0 * matrix1_dim1 * sizeof(float));
    cudaMalloc(&d_matrix2, matrix2_dim0 * matrix2_dim1 * sizeof(float));
    cudaMalloc(&d_output, matrix1_dim0 * matrix1_dim1 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_matrix1, matrix1, matrix1_dim0 * matrix1_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, matrix2_dim0 * matrix2_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((matrix1_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (matrix1_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matadd_kernel<<<numBlocks, threadsPerBlock>>>(
        d_matrix1, d_matrix2, d_output, matrix1_dim0, matrix1_dim1
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, matrix1_dim0 * matrix1_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_output);
}

}  // extern "C"
