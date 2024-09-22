
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);  // Round to nearest even for fp16
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for Laplace filtering using fp16
__global__ void laplace_filter_kernel_fp16(const float* input_tensor, const float* kernel, 
                                        float* output, int batch_size, int channels, int height, int width,
                                        int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        float sum = 0.0f;
        for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i) {
            for (int j = -kernel_size / 2; j <= kernel_size / 2; ++j) {
                int input_row = row + i;
                int input_col = col + j;

                if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                    int input_idx = (input_row * width + input_col) * channels;
                    int kernel_idx = (i + kernel_size / 2) * kernel_size + (j + kernel_size / 2);

                    half input_value = float_to_half(input_tensor[input_idx]);
                    half kernel_value = float_to_half(kernel[kernel_idx]);
                    sum += half_to_float(__hmul(input_value, kernel_value));
                }
            }
        }
        output[row * width + col] = sum;
    }
}

extern "C" {

void laplace_filter_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int channels = va_arg(args, int);
    int height = va_arg(args, int);
    int width = va_arg(args, int);

    // Extract kernel tensor
    const float* kernel = va_arg(args, const float*);
    int kernel_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    laplace_filter_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_kernel, d_output, batch_size, channels, height, width, kernel_size
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

} // extern "C"
