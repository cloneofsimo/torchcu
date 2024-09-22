
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for Roberts cross-gradient edge detection
__global__ void roberts_cross_gradient_kernel_fp32(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        float gx = input[(y + 1) * width + x] - input[y * width + (x - 1)];
        float gy = input[y * width + (x + 1)] - input[(y - 1) * width + x];
        output[y * width + x] = sqrtf(gx * gx + gy * gy);
    } else {
        output[y * width + x] = 0.0f;
    }
}

extern "C" {

void roberts_cross_gradient_fp32(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input = va_arg(args, const float*);
    int width = va_arg(args, int);
    int height = va_arg(args, int);

    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    roberts_cross_gradient_kernel_fp32<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);

    // Copy result back to host
    cudaMemcpy(output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
