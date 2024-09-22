
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

// CUDA kernel for the soft shrink and cross-fade operation
__global__ void softshrink_crossfade_kernel(const float* input_tensor, float threshold, const float* weight, float alpha, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float value = input_tensor[i];
        if (abs(value) > threshold) {
            output[i] = (value - threshold * (value > 0.0f ? 1.0f : -1.0f)) * (1 - alpha) + weight[i] * alpha;
        } else {
            output[i] = weight[i] * alpha;
        }
    }
}

extern "C" {

void softshrink_crossfade_fp32(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract threshold
    float threshold = va_arg(args, float);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract alpha
    float alpha = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int size = input_tensor_dim0 * input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_weight, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    softshrink_crossfade_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, threshold, d_weight, alpha, d_output, size
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
