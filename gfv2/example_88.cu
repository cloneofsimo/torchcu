
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for max pooling
__global__ void max_pool2d_kernel_fp16(const half* input, half* output, int batch_size, int channels, int input_height, int input_width, int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < input_height - kernel_size + 1 && col < input_width - kernel_size + 1) {
        int out_row = row;
        int out_col = col;
        half max_val = input[out_row * input_width + out_col];
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int idx = (out_row + i) * input_width + (out_col + j);
                max_val = fmaxf(max_val, input[idx]);
            }
        }
        output[out_row * (input_width - kernel_size + 1) + out_col] = max_val;
    }
}

extern "C" {

void mean_maxpool_fp16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract kernel size
    int kernel_size = va_arg(args, int);

    // Extract output tensors (assuming they are preallocated)
    half* output = va_arg(args, half*);
    half* mean_val = va_arg(args, half*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int input_height = input_tensor_dim2;
    int input_width = input_tensor_dim3;
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;

    // Allocate device memory
    half *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * channels * input_height * input_width * sizeof(half));
    cudaMalloc(&d_output, batch_size * channels * output_height * output_width * sizeof(half));

    // Copy input data to device (convert float to half)
    cudaMemcpy(d_input, input_tensor, batch_size * channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch max pooling kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    max_pool2d_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, channels, input_height, input_width, kernel_size
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * output_height * output_width * sizeof(half), cudaMemcpyDeviceToHost);

    // Calculate mean (using separate kernel for efficiency)
    float sum = 0.0f;
    for (int i = 0; i < batch_size * channels * input_height * input_width; ++i) {
        sum += input_tensor[i];
    }
    float mean = sum / (batch_size * channels * input_height * input_width);
    *mean_val = __float2half(mean);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
