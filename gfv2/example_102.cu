
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void adaptive_avg_pool2d_kernel(const float* input, float* output, 
                                           int batch_size, int input_channels, int input_height, int input_width,
                                           int output_height, int output_width) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int output_y = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && channel_idx < input_channels && output_y < output_height) {
        float sum = 0.0f;
        int input_y_start = output_y * input_height / output_height;
        int input_y_end = (output_y + 1) * input_height / output_height;
        int input_x_start = 0;
        int input_x_end = input_width;

        for (int input_y = input_y_start; input_y < input_y_end; ++input_y) {
            for (int input_x = input_x_start; input_x < input_x_end; ++input_x) {
                sum += input[batch_idx * input_channels * input_height * input_width +
                             channel_idx * input_height * input_width +
                             input_y * input_width + input_x];
            }
        }

        output[batch_idx * input_channels * output_height * output_width +
               channel_idx * output_height * output_width +
               output_y * output_width] = sum / (input_y_end - input_y_start) / input_width;
    }
}

extern "C" {

void adaptive_avg_pool_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int input_channels = va_arg(args, int);
    int input_height = va_arg(args, int);
    int input_width = va_arg(args, int);

    // Extract output size
    int output_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Calculate output dimensions
    int output_height = output_size;
    int output_width = output_size;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_channels * output_height * output_width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * input_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((output_height + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    adaptive_avg_pool2d_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, input_channels, input_height, input_width,
        output_height, output_width
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_channels * output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
