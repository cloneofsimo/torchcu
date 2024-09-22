
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

// CUDA kernel for matrix transpose using bfloat16
__global__ void matrix_transpose_kernel_bf16(const float* input_matrix, float* output_matrix, int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_cols && col < num_rows) {
        float sum = 0.0f;
        for (int i = 0; i < num_rows; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input_matrix[col * num_rows + i]);
            sum += bfloat16_to_float(__hmul(a, a));  // Transpose operation
        }
        output_matrix[row * num_cols + col] = sum;
    }
}

extern "C" {

void matrix_transpose(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_matrix = va_arg(args, const float*);
    int num_rows = va_arg(args, int);
    int num_cols = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output_matrix = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, num_rows * num_cols * sizeof(float));
    cudaMalloc(&d_output, num_rows * num_cols * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_matrix, num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (num_cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, num_rows, num_cols
    );

    // Copy result back to host
    cudaMemcpy(output_matrix, d_output, num_rows * num_cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
