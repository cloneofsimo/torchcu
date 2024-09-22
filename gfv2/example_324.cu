
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

// CUDA kernel for Wasserstein loss computation using bfloat16
__global__ void wasserstein_loss_kernel_bf16(const float* input_tensor, const float* target_tensor, 
                                             float* output, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        __nv_bfloat16 a = float_to_bfloat16(input_tensor[idx]);
        __nv_bfloat16 b = float_to_bfloat16(target_tensor[idx]);
        output[idx] = bfloat16_to_float(__fabs(a - b));
    }
}

extern "C" {

void wasserstein_bf16_loss(int num_args, ...) {
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

    int batch_size = input_tensor_dim0;

    // Allocate device memory
    float *d_input, *d_target, *d_output;
    cudaMalloc(&d_input, batch_size * sizeof(float));
    cudaMalloc(&d_target, batch_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, batch_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(128);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    wasserstein_loss_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_target, d_output, batch_size
    );

    // Sum the result on the device for mean computation
    cudaDeviceSynchronize();
    float sum_output = 0.0f;
    cudaMemcpy(&sum_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Compute mean on the host and store in the output tensor
    output[0] = sum_output / batch_size;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_output);
}

}  // extern "C"
