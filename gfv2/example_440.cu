
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

// CUDA kernel for calculating L1 loss using bfloat16
__global__ void l1_loss_bf16_kernel(const float* input, const float* target, float* output, 
                                    int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        __nv_bfloat16 a = float_to_bfloat16(input[i]);
        __nv_bfloat16 b = float_to_bfloat16(target[i]);
        output[0] += bfloat16_to_float(abs(a - b)); 
    }
}

extern "C" {

void wasserstein_loss_bf16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int input_size = va_arg(args, int);

    // Extract target tensor
    const float* target = va_arg(args, const float*);
    int target_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Ensure input and target have the same size
    if (input_size != target_size) {
        return; // Or handle the error differently
    }

    // Allocate device memory
    float *d_input, *d_target, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_target, target_size * sizeof(float));
    cudaMalloc(&d_output, sizeof(float)); // Allocate for single output value

    // Copy input data to device
    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, target_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((input_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    l1_loss_bf16_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_target, d_output, input_size);

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_output);
}

} // extern "C"
