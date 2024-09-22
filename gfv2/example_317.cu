
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper functions for converting float to half and vice versa
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for fused sigmoid focal loss
__global__ void fused_sigmoid_focal_loss_kernel(const float* input_tensor, const float* target_tensor, 
                                                float alpha, float gamma, float* output, 
                                                int batch_size, int channels, int height, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < batch_size * channels * height * width) {
        int b = index / (channels * height * width);
        int c = (index % (channels * height * width)) / (height * width);
        int h = ((index % (channels * height * width)) % (height * width)) / width;
        int w = (index % (channels * height * width)) % width;

        // Load input and target values as half-precision
        half input = float_to_half(input_tensor[b * channels * height * width + c * height * width + h * width + w]);
        half target = float_to_half(target_tensor[b * channels * height * width + c * height * width + h * width + w]);

        // Sigmoid activation
        half p = __expf(input) / (1.0f + __expf(input));

        // Focal loss calculation
        half loss = -alpha * (1.0f - p) * (1.0f - p) * target * __logf(p) - (1.0f - alpha) * p * p * (1.0f - target) * __logf(1.0f - p);

        // Store the loss in the output tensor
        output[index] = half_to_float(loss);
    }
}

extern "C" {

void fused_sigmoid_focal_loss_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract target tensor
    const float* target_tensor = va_arg(args, const float*);
    int target_tensor_dim0 = va_arg(args, int);
    int target_tensor_dim1 = va_arg(args, int);
    int target_tensor_dim2 = va_arg(args, int);
    int target_tensor_dim3 = va_arg(args, int);

    // Extract alpha
    float alpha = va_arg(args, float);

    // Extract gamma
    float gamma = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;

    // Allocate device memory
    float* d_input, *d_target, *d_output;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_target, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (batch_size * channels * height * width + threadsPerBlock - 1) / threadsPerBlock;

    fused_sigmoid_focal_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_target, alpha, gamma, d_output, 
        batch_size, channels, height, width
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_output);
}

} // extern "C"
