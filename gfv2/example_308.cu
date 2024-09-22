
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

// CUDA kernel for sigmoid, power, and expand
__global__ void sigmoid_pow_expand_kernel_bf16(const float* input_tensor, float exponent, float* output,
                                        int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        __nv_bfloat16 input_bf16 = float_to_bfloat16(input_tensor[row * n + col]);
        __nv_bfloat16 sigmoid_bf16 = __expf(float_to_bfloat16(-input_bf16)) / (float_to_bfloat16(1.0f) + __expf(float_to_bfloat16(-input_bf16)));
        __nv_bfloat16 powered_bf16 = __powf(sigmoid_bf16, float_to_bfloat16(exponent));
        output[row * n * 5 + col * 5] = bfloat16_to_float(powered_bf16);
        output[row * n * 5 + col * 5 + 1] = bfloat16_to_float(powered_bf16);
        output[row * n * 5 + col * 5 + 2] = bfloat16_to_float(powered_bf16);
        output[row * n * 5 + col * 5 + 3] = bfloat16_to_float(powered_bf16);
        output[row * n * 5 + col * 5 + 4] = bfloat16_to_float(powered_bf16);
    }
}

extern "C" {

void sigmoid_pow_expand_bf16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract exponent
    float exponent = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
    cudaMalloc(&d_output, input_tensor_dim0 * input_tensor_dim1 * 5 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((input_tensor_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_tensor_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sigmoid_pow_expand_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, exponent, d_output, input_tensor_dim0, input_tensor_dim1
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor_dim0 * input_tensor_dim1 * 5 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
