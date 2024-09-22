
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

// CUDA kernel for MSE loss calculation using bfloat16
__global__ void mse_loss_kernel_bf16(const float* input_tensor, const float* target_tensor, float* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        __nv_bfloat16 a = float_to_bfloat16(input_tensor[idx]);
        __nv_bfloat16 b = float_to_bfloat16(target_tensor[idx]);
        __nv_bfloat16 diff = a - b;
        __nv_bfloat16 squared_diff = diff * diff;
        atomicAdd(loss, bfloat16_to_float(squared_diff));
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_size = va_arg(args, int);

    const float* target_tensor = va_arg(args, const float*);
    int target_tensor_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* loss = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_target, *d_loss;
    cudaMalloc(&d_input, input_tensor_size * sizeof(float));
    cudaMalloc(&d_target, target_tensor_size * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, target_tensor_size * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize loss on device
    cudaMemset(d_loss, 0, sizeof(float));

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((input_tensor_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    mse_loss_kernel_bf16<<<numBlocks, threadsPerBlock>>>(d_input, d_target, d_loss, input_tensor_size);

    // Copy result back to host
    cudaMemcpy(loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate final loss
    loss[0] /= input_tensor_size;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_loss);
}

}  // extern "C"
