
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

// CUDA kernel for calculating L1 and MSE loss
__global__ void loss_kernel(const float* input_tensor, const float* target_tensor, float* l1_loss, float* mse_loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float diff = input_tensor[idx] - target_tensor[idx];
        atomicAdd(l1_loss, abs(diff));
        atomicAdd(mse_loss, diff * diff);
    }
}

extern "C" {

void loss_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);

    // Extract target tensor
    const float* target_tensor = va_arg(args, const float*);
    int target_tensor_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int size = input_tensor_dim0;

    // Allocate device memory
    float *d_input, *d_target, *d_l1_loss, *d_mse_loss;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_target, size * sizeof(float));
    cudaMalloc(&d_l1_loss, sizeof(float));
    cudaMalloc(&d_mse_loss, sizeof(float));

    // Initialize loss variables on device
    cudaMemset(d_l1_loss, 0, sizeof(float));
    cudaMemset(d_mse_loss, 0, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    loss_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_target, d_l1_loss, d_mse_loss, size);

    // Copy results back to host
    cudaMemcpy(output, d_l1_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output + 1, d_mse_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_l1_loss);
    cudaFree(d_mse_loss);
}

}  // extern "C"
