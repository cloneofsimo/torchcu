
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for simple convolution
__global__ void simple_convolution_kernel(const float* input_tensor, const float* kernel, float* output_tensor, 
                                          int input_height, int input_width, int input_channels,
                                          int kernel_height, int kernel_width, int kernel_channels) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < input_height && j < input_width) {
        for (int k = 0; k < input_channels; ++k) {
            for (int m = 0; m < kernel_height; ++m) {
                for (int n = 0; n < kernel_width; ++n) {
                    for (int c = 0; c < kernel_channels; ++c) {
                        atomicAdd(&output_tensor[(i * input_width + j) * kernel_channels + c], 
                                  input_tensor[(i * input_width + j) * input_channels + k] * 
                                  kernel[(m * kernel_width + n) * kernel_channels + c]);
                    }
                }
            }
        }
    }
}

extern "C" {

void simple_convolution(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract kernel tensor
    const float* kernel = va_arg(args, const float*);
    int kernel_dim0 = va_arg(args, int);
    int kernel_dim1 = va_arg(args, int);
    int kernel_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output_tensor = va_arg(args, float*);

    va_end(args);

    int input_height = input_tensor_dim0;
    int input_width = input_tensor_dim1;
    int input_channels = input_tensor_dim2;

    int kernel_height = kernel_dim0;
    int kernel_width = kernel_dim1;
    int kernel_channels = kernel_dim2;

    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_height * input_width * input_channels * sizeof(float));
    cudaMalloc(&d_kernel, kernel_height * kernel_width * kernel_channels * sizeof(float));
    cudaMalloc(&d_output, input_height * input_width * kernel_channels * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_height * input_width * input_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_height * kernel_width * kernel_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize output tensor on device
    cudaMemset(d_output, 0, input_height * input_width * kernel_channels * sizeof(float));

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

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
