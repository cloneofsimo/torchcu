
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

// CUDA kernel for where operation with bfloat16 and transpose
__global__ void where_transpose_bf16_kernel(const float* input_tensor, const bool* condition, const float* other, float* output, 
                                        int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        __nv_bfloat16 a = float_to_bfloat16(input_tensor[row * n + col]);
        __nv_bfloat16 b = float_to_bfloat16(other[row * n + col]);
        output[col * m + row] = bfloat16_to_float(condition[row * n + col] ? a : b); // Transpose access
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    const bool* condition = va_arg(args, const bool*);
    int condition_dim0 = va_arg(args, int);
    int condition_dim1 = va_arg(args, int);

    const float* other = va_arg(args, const float*);
    int other_dim0 = va_arg(args, int);
    int other_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Check dimensions are compatible
    if (input_tensor_dim0 != condition_dim0 || input_tensor_dim1 != condition_dim1 || 
        input_tensor_dim0 != other_dim0 || input_tensor_dim1 != other_dim1) {
        // Handle dimension mismatch error (e.g., throw an exception)
        return;
    }

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_other, *d_output;
    bool *d_condition;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_other, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_condition, batch_size * input_dim * sizeof(bool));
    cudaMalloc(&d_output, input_dim * batch_size * sizeof(float)); // Transposed output

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_other, other, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_condition, condition, batch_size * input_dim * sizeof(bool), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    where_transpose_bf16_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_condition, d_other, d_output, batch_size, input_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, input_dim * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_other);
    cudaFree(d_condition);
    cudaFree(d_output);
}

}  // extern "C"
