
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for pixel shuffle and hardsigmoid activation
__global__ void pixel_shuffle_hardsigmoid_kernel(const float* input, float* output, int batch_size, int in_channels, 
                                                int in_height, int in_width, int upscale_factor) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.z * blockDim.z + threadIdx.z;
    int w = threadIdx.w;

    if (b < batch_size && c < in_channels && h < in_height && w < in_width) {
        // Calculate the original index in the input tensor
        int original_c = c / (upscale_factor * upscale_factor);
        int original_h = h * upscale_factor;
        int original_w = w * upscale_factor;
        int original_index = (b * in_channels * in_height * in_width) + 
                            (original_c * in_height * in_width) + 
                            (original_h * in_width) + original_w;

        // Calculate the output index
        int output_index = (b * in_channels * in_height * in_width) + 
                           (c * in_height * in_width) + 
                           (h * in_width) + w;

        // Apply hardsigmoid activation
        float value = input[original_index];
        output[output_index] = (value + 1.0f) / 6.0f; // hardsigmoid(x) = max(0, min(1, (x + 1) / 6))
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int in_channels = va_arg(args, int);
    int in_height = va_arg(args, int);
    int in_width = va_arg(args, int);

    // Extract upscale factor
    int upscale_factor = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Calculate output dimensions
    int out_channels = in_channels / (upscale_factor * upscale_factor);
    int out_height = in_height * upscale_factor;
    int out_width = in_width * upscale_factor;

    // Allocate device memory
    float* d_input, *d_output;
    cudaMalloc(&d_input, batch_size * in_channels * in_height * in_width * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * out_height * out_width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * in_channels * in_height * in_width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1, 1);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (in_channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (in_height + threadsPerBlock.z - 1) / threadsPerBlock.z,
                   (in_width + threadsPerBlock.w - 1) / threadsPerBlock.w);
    pixel_shuffle_hardsigmoid_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, in_channels, in_height, in_width, upscale_factor
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * out_channels * out_height * out_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
