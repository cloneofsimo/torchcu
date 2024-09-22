
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for adaptive average pooling
__global__ void adaptive_avg_pool2d_kernel(const float* input, float* output, int batch_size, int channels, 
                                            int input_height, int input_width, int output_height, int output_width) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int output_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && channel_idx < channels && output_row < output_height) {
        float sum = 0.0f;
        int input_rows_start = output_row * input_height / output_height;
        int input_rows_end = (output_row + 1) * input_height / output_height;
        int input_cols_start = 0;
        int input_cols_end = input_width;

        for (int i = input_rows_start; i < input_rows_end; ++i) {
            for (int j = input_cols_start; j < input_cols_end; ++j) {
                sum += input[batch_idx * channels * input_height * input_width + 
                           channel_idx * input_height * input_width + i * input_width + j];
            }
        }
        output[batch_idx * channels * output_height * output_width + 
               channel_idx * output_height * output_width + output_row] = sum / (input_height * input_width);
    }
}

extern "C" {

void identity_adaptive_avg_pool2d(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract output_size
    int output_size = va_arg(args, int);

    // Extract inplace flag (as bool)
    bool inplace = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int input_height = input_tensor_dim2;
    int input_width = input_tensor_dim3;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * output_size * output_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);

    if (!inplace) {
        // Launch kernel for adaptive average pooling
        dim3 threadsPerBlock(16, 16, 1);
        dim3 numBlocks((output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

        adaptive_avg_pool2d_kernel<<<numBlocks, threadsPerBlock>>>(
            d_input, d_output, batch_size, channels, input_height, input_width, output_size, output_size
        );
    } else {
        // In-place operation, just copy input to output
        cudaMemcpy(d_output, d_input, batch_size * channels * input_height * input_width * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
