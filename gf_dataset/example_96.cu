
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper function to perform reflection padding
__global__ void reflect_padding(const float* input, float* padded_input,
                                 int batch, int channels, int height, int width,
                                 int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height + 2 * (kernel_size / 2) && col < width + 2 * (kernel_size / 2)) {
        int padded_row = row;
        int padded_col = col;

        // Handle reflection padding
        if (row < kernel_size / 2) {
            padded_row = kernel_size / 2 - 1 - (row - kernel_size / 2);
        } else if (row >= height + kernel_size / 2) {
            padded_row = height - 1 - (row - (height + kernel_size / 2 - 1));
        }

        if (col < kernel_size / 2) {
            padded_col = kernel_size / 2 - 1 - (col - kernel_size / 2);
        } else if (col >= width + kernel_size / 2) {
            padded_col = width - 1 - (col - (width + kernel_size / 2 - 1));
        }

        padded_input[row * (width + 2 * (kernel_size / 2)) + col] =
                input[padded_row * width + padded_col];
    }
}

// CUDA kernel for 2D convolution with reflection padding
__global__ void conv2d_kernel(const float* padded_input, const float* kernel, float* output,
                                int batch, int channels, int height, int width,
                                int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    sum += padded_input[(row + kh) * (width + 2 * (kernel_size / 2)) + (col + kw)] *
                          kernel[c * kernel_size * kernel_size + kh * kernel_size + kw];
                }
            }
            output[row * width + col + c * height * width] = sum;
        }
    }
}

extern "C" {
    // Function signature for calling the CUDA kernel from Python
    void torch_function(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract input tensors and shapes
        const float* input_tensor = va_arg(args, const float*);
        int input_tensor_batch = va_arg(args, int);
        int input_tensor_channels = va_arg(args, int);
        int input_tensor_height = va_arg(args, int);
        int input_tensor_width = va_arg(args, int);

        const float* kernel = va_arg(args, const float*);
        int kernel_size = va_arg(args, int);

        // Extract output tensor (assuming it's preallocated)
        float* output = va_arg(args, float*);

        va_end(args);

        // Calculate padded dimensions
        int padded_height = input_tensor_height + 2 * (kernel_size / 2);
        int padded_width = input_tensor_width + 2 * (kernel_size / 2);

        // Allocate device memory for input, padded input, and output
        float *d_input, *d_padded_input, *d_output;
        cudaMalloc(&d_input, input_tensor_batch * input_tensor_channels * input_tensor_height * input_tensor_width * sizeof(float));
        cudaMalloc(&d_padded_input, input_tensor_batch * input_tensor_channels * padded_height * padded_width * sizeof(float));
        cudaMalloc(&d_output, input_tensor_batch * input_tensor_channels * input_tensor_height * input_tensor_width * sizeof(float));

        // Copy input data to device
        cudaMemcpy(d_input, input_tensor, input_tensor_batch * input_tensor_channels * input_tensor_height * input_tensor_width * sizeof(float), cudaMemcpyHostToDevice);

        // Launch reflection padding kernel
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((padded_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (padded_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        reflect_padding<<<numBlocks, threadsPerBlock>>>(
            d_input, d_padded_input, input_tensor_batch, input_tensor_channels, input_tensor_height, input_tensor_width, kernel_size);

        // Launch convolution kernel
        threadsPerBlock = dim3(16, 16);
        numBlocks = dim3((input_tensor_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (input_tensor_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        conv2d_kernel<<<numBlocks, threadsPerBlock>>>(
            d_padded_input, kernel, d_output, input_tensor_batch, input_tensor_channels, input_tensor_height, input_tensor_width, kernel_size);

        // Copy result back to host
        cudaMemcpy(output, d_output, input_tensor_batch * input_tensor_channels * input_tensor_height * input_tensor_width * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_padded_input);
        cudaFree(d_output);
    }
}
