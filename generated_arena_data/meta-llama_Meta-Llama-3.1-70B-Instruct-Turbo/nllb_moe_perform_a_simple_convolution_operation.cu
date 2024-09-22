
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for simple convolution
__global__ void simple_convolution_kernel(const float* input_tensor, const float* kernel, float* output_tensor, 
                                        int input_height, int input_width, int input_channels, 
                                        int kernel_height, int kernel_width, int kernel_channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < input_height && y < input_width && z < kernel_channels) {
        float sum = 0.0f;
        for (int i = 0; i < input_channels; ++i) {
            for (int m = 0; m < kernel_height; ++m) {
                for (int n = 0; n < kernel_width; ++n) {
                    sum += input_tensor[(x * input_width + y) * input_channels + i] * kernel[(m * kernel_width + n) * input_channels + i];
                }
            }
        }
        output_tensor[(x * input_width + y) * kernel_channels + z] = sum;
    }
}

extern "C" {

void simple_convolution(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_height = va_arg(args, int);
    int input_width = va_arg(args, int);
    int input_channels = va_arg(args, int);

    // Extract kernel tensor
    const float* kernel = va_arg(args, const float*);
    int kernel_height = va_arg(args, int);
    int kernel_width = va_arg(args, int);
    int kernel_channels = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output_tensor = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_height * input_width * input_channels * sizeof(float));
    cudaMalloc(&d_kernel, kernel_height * kernel_width * input_channels * sizeof(float));
    cudaMalloc(&d_output, input_height * input_width * kernel_channels * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_height * input_width * input_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_height * kernel_width * input_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((input_height + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (input_width + threadsPerBlock.y - 1) / threadsPerBlock.y,
                (kernel_channels + threadsPerBlock.z - 1) / threadsPerBlock.z);

    simple_convolution_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_kernel, d_output, input_height, input_width, input_channels, kernel_height, kernel_width, kernel_channels
    );

    // Copy result back to host
    cudaMemcpy(output_tensor, d_output, input_height * input_width * kernel_channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

}  // extern "C"
