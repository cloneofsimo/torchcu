
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end
#include <cutlass/cutlass.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for CELU activation using bfloat16 with Cutlass
__global__ void celu_kernel_bf16(const float* input_tensor, float* output, 
                                  int m, int n, float alpha) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        __nv_bfloat16 input_bf16 = float_to_bfloat16(input_tensor[row * n + col]);
        __nv_bfloat16 alpha_bf16 = float_to_bfloat16(alpha);

        __nv_bfloat16 result = (input_bf16 > 0.0f) ? input_bf16 : alpha_bf16 * (exp(input_bf16 / alpha_bf16) - 1.0f);

        output[row * n + col] = bfloat16_to_float(result);
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract alpha
    float alpha = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    celu_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, input_dim, alpha
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
