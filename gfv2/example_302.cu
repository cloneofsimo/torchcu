
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h> 

__global__ void depthwise_separable_conv_int8_kernel(const int8_t* input, const int8_t* depthwise_weight, const int8_t* pointwise_weight, int8_t* output,
                                                   int batch_size, int input_channels, int output_channels, 
                                                   int input_height, int input_width, int kernel_size, int stride) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    int output_x = blockIdx.z * blockDim.z + threadIdx.z;

    if (batch < batch_size && output_y < input_height && output_x < input_width) {
        int output_index = batch * output_channels * input_height * input_width + output_y * input_width * output_channels + output_x * output_channels;

        for (int out_channel = 0; out_channel < output_channels; ++out_channel) {
            int sum = 0;
            for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
                int input_index = batch * input_channels * input_height * input_width + 
                                  output_y * input_width * input_channels + output_x * input_channels + in_channel;
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int kernel_index = out_channel * input_channels * kernel_size * kernel_size + 
                                          ky * kernel_size * input_channels + kx * input_channels + in_channel;

                        int input_y = output_y * stride + ky - kernel_size / 2;
                        int input_x = output_x * stride + kx - kernel_size / 2;

                        if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                            sum += input[input_index + (input_y * input_width + input_x) * input_channels] * depthwise_weight[kernel_index];
                        }
                    }
                }
            }
            output[output_index + out_channel] = sum * pointwise_weight[out_channel];
        }
    }
}

extern "C" {

void depthwise_separable_conv_int8(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int input_channels = va_arg(args, int);
    int input_height = va_arg(args, int);
    int input_width = va_arg(args, int);

    // Extract depthwise weight tensor
    const float* depthwise_weight = va_arg(args, const float*);
    int depthwise_weight_channels = va_arg(args, int);
    int depthwise_weight_height = va_arg(args, int);
    int depthwise_weight_width = va_arg(args, int);

    // Extract pointwise weight tensor
    const float* pointwise_weight = va_arg(args, const float*);
    int pointwise_weight_channels = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory for int8 data
    int8_t* d_input;
    int8_t* d_depthwise_weight;
    int8_t* d_pointwise_weight;
    int8_t* d_output;
    cudaMalloc(&d_input, batch_size * input_channels * input_height * input_width * sizeof(int8_t));
    cudaMalloc(&d_depthwise_weight, depthwise_weight_channels * depthwise_weight_height * depthwise_weight_width * sizeof(int8_t));
    cudaMalloc(&d_pointwise_weight, pointwise_weight_channels * sizeof(int8_t));
    cudaMalloc(&d_output, batch_size * pointwise_weight_channels * input_height * input_width * sizeof(int8_t));

    // Copy data to device, quantizing to int8
    cudaMemcpy(d_input, input_tensor, batch_size * input_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_depthwise_weight, depthwise_weight, depthwise_weight_channels * depthwise_weight_height * depthwise_weight_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointwise_weight, pointwise_weight, pointwise_weight_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int kernel_size = depthwise_weight_height;
    int stride = 1;
    dim3 threadsPerBlock(8, 8, 8); 
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (input_width + threadsPerBlock.z - 1) / threadsPerBlock.z);

    depthwise_separable_conv_int8_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_depthwise_weight, d_pointwise_weight, d_output, 
        batch_size, input_channels, pointwise_weight_channels,
        input_height, input_width, kernel_size, stride
    );

    // Copy results back to host, dequantizing to fp32
    cudaMemcpy(output, d_output, batch_size * pointwise_weight_channels * input_height * input_width * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_depthwise_weight);
    cudaFree(d_pointwise_weight);
    cudaFree(d_output);
}

}  // extern "C"
