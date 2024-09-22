
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for pixel shuffle and unsqueeze
__global__ void pixel_shuffle_unsqueeze_kernel(const float* input_tensor, float* output, 
                                        int batch_size, int channels, int height, int width, int upscale_factor) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < batch_size && y < height * upscale_factor && x < width * upscale_factor) {
        int c = (y / upscale_factor) * width + (x / upscale_factor);
        int oy = y % upscale_factor;
        int ox = x % upscale_factor;
        int output_idx = b * channels * height * width * upscale_factor * upscale_factor + 
                         oy * width * upscale_factor + ox + 
                         c * upscale_factor * upscale_factor;
        
        int input_idx = b * channels * height * width + c + (y / upscale_factor) * width * upscale_factor + (x / upscale_factor);
        output[output_idx] = input_tensor[input_idx];
    }
}

extern "C" {

void pixel_shuffle_unsqueeze(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int channels = va_arg(args, int);
    int height = va_arg(args, int);
    int width = va_arg(args, int);

    // Extract upscale factor
    int upscale_factor = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * upscale_factor * upscale_factor * height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height * upscale_factor + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (width * upscale_factor + threadsPerBlock.z - 1) / threadsPerBlock.z);

    pixel_shuffle_unsqueeze_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, channels, height, width, upscale_factor
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * upscale_factor * upscale_factor * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
