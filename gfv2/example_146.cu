
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for calculating image gradients using FP16
__global__ void image_gradient_kernel_fp16(const half* image, half* gradient, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width - 1 && y < height - 1) {
        // Calculate horizontal gradient
        gradient[0 * height * width + y * width + x] = image[(y + 0) * width + (x + 1)] - image[(y + 0) * width + x];
        // Calculate vertical gradient
        gradient[1 * height * width + y * width + x] = image[(y + 1) * width + (x + 0)] - image[y * width + (x + 0)];
    }
}

extern "C" {

void image_gradient_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* image = va_arg(args, const float*);
    int height = va_arg(args, int);
    int width = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* gradient = va_arg(args, float*);

    va_end(args);

    // Allocate device memory for image and gradient
    half *d_image, *d_gradient;
    cudaMalloc(&d_image, height * width * sizeof(half));
    cudaMalloc(&d_gradient, 6 * height * width * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_image, image, height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    image_gradient_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_image, d_gradient, height, width
    );

    // Copy result back to host
    cudaMemcpy(gradient, d_gradient, 6 * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_gradient);
}

}  // extern "C"
