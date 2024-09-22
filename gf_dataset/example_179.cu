
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdarg.h>
#include <cutlass/cutlass.h>

// CUDA kernel for Instance Normalization followed by ReLU
__global__ void instance_norm_relu_kernel(const float* input, const float* weight, const float* bias, float* output,
                                            int batch_size, int channels, int height, int width) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int h_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (batch_idx < batch_size && channel_idx < channels && h_idx < height) {
        // Calculate the index of the current element in the input tensor
        int input_idx = batch_idx * channels * height * width + channel_idx * height * width + h_idx * width;

        // Calculate the index of the current element in the weight tensor
        int weight_idx = channel_idx;

        // Calculate the index of the current element in the bias tensor
        int bias_idx = channel_idx;

        float input_value = input[input_idx];
        float weight_value = weight[weight_idx];
        float bias_value = bias[bias_idx];

        // Calculate the mean and variance of the current channel across the spatial dimensions
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int w_idx = 0; w_idx < width; w_idx++) {
            int idx = input_idx + w_idx;
            sum += input[idx];
            sum_sq += input[idx] * input[idx];
        }

        float mean = sum / (width * height);
        float variance = sum_sq / (width * height) - mean * mean;

        // Calculate the normalized value
        float normalized_value = (input_value - mean) / sqrt(variance + 1e-5);

        // Multiply by the weight and add the bias
        float output_value = normalized_value * weight_value + bias_value;

        // Apply ReLU activation
        output_value = fmaxf(output_value, 0.0f);

        // Store the output value
        output[input_idx] = output_value;
    }
}

// Helper function to allocate device memory
void *alloc_device_memory(size_t size) {
    void *ptr;
    cudaMalloc(&ptr, size);
    if (ptr == NULL) {
        std::cerr << "Failed to allocate device memory!" << std::endl;
        exit(1);
    }
    return ptr;
}

// Helper function to copy data to device
void copy_to_device(const void *src, void *dst, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "Failed to copy data to device!" << std::endl;
        exit(1);
    }
}

// Helper function to copy data from device
void copy_from_device(const void *src, void *dst, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "Failed to copy data from device!" << std::endl;
        exit(1);
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int channels = va_arg(args, int);
    int height = va_arg(args, int);
    int width = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_size = va_arg(args, int);

    const float* bias = va_arg(args, const float*);
    int bias_size = va_arg(args, int);

    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory for input, weight, bias, and output
    float *d_input = reinterpret_cast<float*>(alloc_device_memory(batch_size * channels * height * width * sizeof(float)));
    float *d_weight = reinterpret_cast<float*>(alloc_device_memory(weight_size * sizeof(float)));
    float *d_bias = reinterpret_cast<float*>(alloc_device_memory(bias_size * sizeof(float)));
    float *d_output = reinterpret_cast<float*>(alloc_device_memory(batch_size * channels * height * width * sizeof(float)));

    // Copy data from host to device
    copy_to_device(input, d_input, batch_size * channels * height * width * sizeof(float));
    copy_to_device(weight, d_weight, weight_size * sizeof(float));
    copy_to_device(bias, d_bias, bias_size * sizeof(float));

    // Launch the kernel
    dim3 block_size(16, 16, 16);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x, (channels + block_size.y - 1) / block_size.y, (height + block_size.z - 1) / block_size.z);
    instance_norm_relu_kernel<<<grid_size, block_size>>>(d_input, d_weight, d_bias, d_output, batch_size, channels, height, width);

    // Copy data from device to host
    copy_from_device(d_output, output, batch_size * channels * height * width * sizeof(float));

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}
