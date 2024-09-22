
#include <cuda_runtime.h>
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

// CUDA kernel for simple convolution using bfloat16
__global__ void simple_convolution_kernel_bf16(const float* input_tensor, const float* kernel, float* output, 
                                            int input_height, int input_width, int input_channels, 
                                            int kernel_height, int kernel_width, int kernel_channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < input_height && col < input_width) {
        float sum = 0.0f;
        for (int i = 0; i < input_channels; ++i) {
            for (int m = 0; m < kernel_height; ++m) {
                for (int n = 0; n < kernel_width; ++n) {
                    __nv_bfloat16 a = float_to_bfloat16(input_tensor[row * input_width * input_channels + col * input_channels + i]);
                    __nv_bfloat16 b = float_to_bfloat16(kernel[m * kernel_width * kernel_channels + n * kernel_channels + i]);
                    sum += bfloat16_to_float(__hmul(a, b));
                }
            }
        }
        output[row * input_width * kernel_channels + col * kernel_channels] = sum;
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
    float* output = va_arg(args, float*);

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

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_height + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (input_width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    simple_convolution_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_kernel, d_output, input_height, input_width, input_channels, 
        kernel_height, kernel_width, kernel_channels
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, input_height * input_width * kernel_channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

}  // extern "C"
