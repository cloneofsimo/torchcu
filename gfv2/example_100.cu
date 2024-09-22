
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for Mish activation and transpose using bfloat16
__global__ void mish_transpose_kernel_bf16(const float* input_tensor, float* output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < m) {
        __nv_bfloat16 x = float_to_bfloat16(input_tensor[col * n + row]);  // Transpose access
        __nv_bfloat16 y = __hmul(x, __hmul(x, __hmul(x, x - 3.0f) + 3.0f));  // Mish calculation
        output[row * m + col] = bfloat16_to_float(y);
    }
}

extern "C" {

void mish_transpose_bfloat16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int m = input_tensor_dim0;
    int n = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, m * n * sizeof(float));
    cudaMalloc(&d_output, n * m * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    mish_transpose_kernel_bfloat16<<<numBlocks, threadsPerBlock>>>(d_input, d_output, m, n);

    // Copy result back to host
    cudaMemcpy(output, d_output, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
