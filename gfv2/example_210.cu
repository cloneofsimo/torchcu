
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

// CUDA kernel for calculating Frobenius norm
__global__ void frobenius_norm_kernel(const float* input_tensor, float* norm, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        atomicAdd(norm, input_tensor[i] * input_tensor[i]);
    }
}

// CUDA kernel for SELU activation (inplace)
__global__ void selu_kernel(float* input_tensor, int size, float scale, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        input_tensor[i] = scale * (input_tensor[i] > 0.0f ? input_tensor[i] : alpha * (expf(input_tensor[i]) - 1.0f));
    }
}

extern "C" {

void frobenius_norm_selu_inplace(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output_norm = va_arg(args, float*);

    // Extract scale and alpha values
    float scale = va_arg(args, float);
    float alpha = va_arg(args, float);

    va_end(args);

    int size = input_tensor_dim0 * input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output_norm;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output_norm, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, size * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate Frobenius norm
    dim3 threadsPerBlock(256);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    frobenius_norm_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output_norm, size);

    // Apply SELU activation inplace
    selu_kernel<<<numBlocks, threadsPerBlock>>>(d_input, size, scale, alpha);

    // Copy norm back to host
    cudaMemcpy(output_norm, d_output_norm, sizeof(float), cudaMemcpyDeviceToHost);

    // Copy modified input back to host (inplace operation)
    cudaMemcpy(input_tensor, d_input, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output_norm);
}

}  // extern "C"
