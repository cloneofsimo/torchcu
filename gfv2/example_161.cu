
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

#define BLOCK_SIZE 16

// Function for calculating softmax with float inputs and int8 outputs
__global__ void softmax_int8(const float* input, int8_t* output, int batch_size, int channels, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = (row * width + col) * channels;
        float sum = 0.0f;

        for (int c = 0; c < channels; ++c) {
            sum += expf(input[idx + c]);
        }

        for (int c = 0; c < channels; ++c) {
            output[idx + c] = __int8_as_float(roundf(expf(input[idx + c]) / sum * 255.0f));
        }
    }
}

// Function for coordinate attention calculation with float inputs
__global__ void coord_attention_kernel(const float* input, float* output, 
                                         int batch_size, int channels, int height, int width, 
                                         int stride) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = (row * width + col) * channels;
        float sum_h = 0.0f, sum_w = 0.0f;

        // Calculate horizontal and vertical attention
        for (int i = 0; i < width; ++i) {
            sum_h += input[idx + (i * channels)];
        }
        sum_h /= width;
        for (int i = 0; i < height; ++i) {
            sum_w += input[idx + (i * width * channels)];
        }
        sum_w /= height;

        float att_h = expf(sum_h) / (expf(sum_h) + expf(sum_w));
        float att_w = expf(sum_w) / (expf(sum_h) + expf(sum_w));

        // Apply attention
        for (int c = 0; c < channels; ++c) {
            output[idx + c] = att_h * input[idx + c] + att_w * input[idx + c];
        }
    }
}

__global__ void transposed_conv2d_int8_kernel(const int8_t* input, const int8_t* weight, const int8_t* bias,
                                              int8_t* output, 
                                              int batch_size, int input_channels, int output_channels,
                                              int input_height, int input_width, int output_height, int output_width,
                                              int kernel_size, int stride, int padding, int output_padding) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_height && col < output_width) {
        for (int oc = 0; oc < output_channels; ++oc) {
            float sum = 0.0f;
            for (int ic = 0; ic < input_channels; ++ic) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int input_row = row * stride - padding + kh;
                        int input_col = col * stride - padding + kw;
                        
                        if (input_row >= 0 && input_row < input_height && input_col >= 0 && input_col < input_width) {
                            int input_idx = (input_row * input_width + input_col) * input_channels + ic;
                            int weight_idx = (oc * kernel_size * kernel_size + kh * kernel_size + kw) * input_channels + ic;
                            sum += __int8_as_float(input[input_idx]) * __int8_as_float(weight[weight_idx]);
                        }
                    }
                }
            }
            sum += __int8_as_float(bias[oc]);
            output[(row * output_width + col) * output_channels + oc] = __int8_as_float(roundf(sum * 255.0f));
        }
    }
}

extern "C" {

void coord_attention_transposed_conv2d_int8(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);
    int weight_dim3 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    // Extract stride
    int stride = va_arg(args, int);

    // Extract padding
    int padding = va_arg(args, int);

    // Extract output_padding
    int output_padding = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_channels = input_tensor_dim1;
    int input_height = input_tensor_dim2;
    int input_width = input_tensor_dim3;

    int output_channels = weight_dim0;
    int kernel_size = weight_dim2;

    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + 2 * output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + 2 * output_padding;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    int8_t *d_output_int8;
    cudaMalloc(&d_input, batch_size * input_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_weight, output_channels * input_channels * kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_bias, output_channels * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_channels * output_height * output_width * sizeof(float));
    cudaMalloc(&d_output_int8, batch_size * output_channels * output_height * output_width * sizeof(int8_t));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_channels * input_channels * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Transposed Convolution (int8)
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((output_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transposed_conv2d_int8_kernel<<<numBlocks, threadsPerBlock>>>(
        reinterpret_cast<const int8_t*>(d_input), reinterpret_cast<const int8_t*>(d_weight), reinterpret_cast<const int8_t*>(d_bias),
        d_output_int8, batch_size, input_channels, output_channels,
        input_height, input_width, output_height, output_width,
        kernel_size, stride, padding, output_padding
    );

    // Coordinate Attention (int8)
    coord_attention_kernel<<<numBlocks, threadsPerBlock>>>(
        reinterpret_cast<float*>(d_output_int8), d_output,
        batch_size, output_channels, output_height, output_width,
        stride
    );

    // Fused Softmax (int8)
    numBlocks = ((output_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    softmax_int8<<<numBlocks, threadsPerBlock>>>(
        d_output, d_output_int8, batch_size, output_channels, output_height, output_width
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_channels * output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaFree(d_output_int8);
}

}  // extern "C"
