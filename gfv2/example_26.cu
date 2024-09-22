
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for max pooling
__global__ void max_pool1d_kernel(const float* input, float* output, int batch_size, int channels, int input_size, int kernel_size, int stride) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch_size && channel_idx < channels) {
        int output_idx = batch_idx * channels * (input_size / stride) + channel_idx * (input_size / stride);

        for (int i = 0; i < input_size / stride; ++i) {
            float max_val = -FLT_MAX;
            for (int j = 0; j < kernel_size; ++j) {
                int input_offset = batch_idx * channels * input_size + channel_idx * input_size + i * stride + j;
                if (input_offset < batch_size * channels * input_size) {
                    max_val = fmaxf(max_val, input[input_offset]);
                }
            }
            output[output_idx + i] = max_val;
        }
    }
}

extern "C" {

void identity_maxpool_fp32_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int input_size = input_tensor_dim2;

    // Define kernel parameters
    int kernel_size = 2;
    int stride = 2;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * channels * input_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * (input_size / stride) * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * channels * input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch max pooling kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (channels + threadsPerBlock.y - 1) / threadsPerBlock.y);

    max_pool1d_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, batch_size, channels, input_size, kernel_size, stride);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * (input_size / stride) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

} // extern "C"
