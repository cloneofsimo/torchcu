
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for convolution
__global__ void conv2d_kernel_fp16(const half* input, const half* weight, const half* bias, half* output,
                                 int batch_size, int input_channels, int output_channels,
                                 int input_height, int input_width, int kernel_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.z * blockDim.z + threadIdx.z;
    int w = threadIdx.x;

    if (b < batch_size && o < output_channels && h < input_height && w < input_width) {
        float sum = 0.0f;
        for (int ic = 0; ic < input_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_h = h + kh - kernel_size / 2;
                    int in_w = w + kw - kernel_size / 2;

                    if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                        sum += __int2float_rn(
                            __hmul(__float2half_rn(input[b * input_channels * input_height * input_width +
                                                   ic * input_height * input_width + in_h * input_width + in_w]),
                                   __float2half_rn(weight[o * input_channels * kernel_size * kernel_size +
                                                   ic * kernel_size * kernel_size + kh * kernel_size + kw])));
                    }
                }
            }
        }
        output[b * output_channels * input_height * input_width + o * input_height * input_width + h * input_width + w] =
            __float2half_rn(sum + __int2float_rn(bias[o]));
    }
}

// CUDA kernel for ReLU activation
__global__ void relu_kernel_fp16(half* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] = __int2half_rn(fmaxf(0.0f, __half2float_rn(data[i])));
    }
}

// CUDA kernel for adaptive max pooling
__global__ void adaptive_max_pool2d_kernel_fp16(const half* input, half* output,
                                                 int batch_size, int channels, int input_height, int input_width) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (b < batch_size && c < channels) {
        float max_val = -INFINITY;
        for (int h = 0; h < input_height; ++h) {
            for (int w = 0; w < input_width; ++w) {
                float val = __half2float_rn(
                    input[b * channels * input_height * input_width + c * input_height * input_width + h * input_width + w]);
                max_val = fmaxf(max_val, val);
            }
        }
        output[b * channels + c] = __float2half_rn(max_val);
    }
}

extern "C" {
    void acoustic_model_function(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract input tensor
        const float* input = va_arg(args, const float*);
        int input_dim0 = va_arg(args, int);
        int input_dim1 = va_arg(args, int);
        int input_dim2 = va_arg(args, int);
        int input_dim3 = va_arg(args, int);

        // Extract weight tensor
        const float* weight = va_arg(args, const float*);
        int weight_dim0 = va_arg(args, int);
        int weight_dim1 = va_arg(args, int);
        int weight_dim2 = va_arg(args, int);
        int weight_dim3 = va_arg(args, int);

        // Extract bias tensor
        const float* bias = va_arg(args, const float*);
        int bias_dim0 = va_arg(args, int);

        // Extract output tensor (assuming it's preallocated)
        float* output = va_arg(args, float*);

        va_end(args);

        // Allocate device memory
        half *d_input, *d_weight, *d_bias, *d_conv_output, *d_pooled_output;
        cudaMalloc(&d_input, input_dim0 * input_dim1 * input_dim2 * input_dim3 * sizeof(half));
        cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * sizeof(half));
        cudaMalloc(&d_bias, bias_dim0 * sizeof(half));
        cudaMalloc(&d_conv_output, input_dim0 * weight_dim0 * input_dim2 * input_dim3 * sizeof(half));
        cudaMalloc(&d_pooled_output, input_dim0 * weight_dim0 * sizeof(half));

        // Copy input data to device
        cudaMemcpy(d_input, input, input_dim0 * input_dim1 * input_dim2 * input_dim3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, bias, bias_dim0 * sizeof(float), cudaMemcpyHostToDevice);

        // Launch convolution kernel
        dim3 conv_threadsPerBlock(16, 16);
        dim3 conv_numBlocks((input_dim3 + conv_threadsPerBlock.x - 1) / conv_threadsPerBlock.x,
                            (weight_dim0 + conv_threadsPerBlock.y - 1) / conv_threadsPerBlock.y,
                            (input_dim2 + conv_threadsPerBlock.z - 1) / conv_threadsPerBlock.z);
        conv2d_kernel_fp16<<<conv_numBlocks, conv_threadsPerBlock>>>(
            d_input, d_weight, d_bias, d_conv_output, input_dim0, input_dim1, weight_dim0, input_dim2, input_dim3, weight_dim2);
        cudaDeviceSynchronize();

        // Launch ReLU kernel
        dim3 relu_threadsPerBlock(256);
        dim3 relu_numBlocks((input_dim0 * weight_dim0 * input_dim2 * input_dim3 + relu_threadsPerBlock.x - 1) / relu_threadsPerBlock.x);
        relu_kernel_fp16<<<relu_numBlocks, relu_threadsPerBlock>>>(d_conv_output, input_dim0 * weight_dim0 * input_dim2 * input_dim3);
        cudaDeviceSynchronize();

        // Launch adaptive max pooling kernel
        dim3 pool_threadsPerBlock(32, 32);
        dim3 pool_numBlocks((input_dim0 + pool_threadsPerBlock.x - 1) / pool_threadsPerBlock.x,
                            (weight_dim0 + pool_threadsPerBlock.y - 1) / pool_threadsPerBlock.y);
        adaptive_max_pool2d_kernel_fp16<<<pool_numBlocks, pool_threadsPerBlock>>>(
            d_conv_output, d_pooled_output, input_dim0, weight_dim0, input_dim2, input_dim3);
        cudaDeviceSynchronize();

        // Copy result back to host
        cudaMemcpy(output, d_pooled_output, input_dim0 * weight_dim0 * sizeof(half), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_weight);
        cudaFree(d_bias);
        cudaFree(d_conv_output);
        cudaFree(d_pooled_output);
    }
}
