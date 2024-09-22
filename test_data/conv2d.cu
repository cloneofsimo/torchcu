#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end
#include <stdio.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for 2D convolution and ReLU using bfloat16
__global__ void conv2d_relu_kernel_bf16(const float* input_tensor, const float* weight, const float* bias, 
                                        float* output, int batch_size, int in_channels, int out_channels,
                                        int input_height, int input_width, int kernel_height, int kernel_width) {
    int batch = blockIdx.z;
    int out_channel = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = threadIdx.y;

    if (row < input_height && col < input_width) {
        float sum = 0.0f;

        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int input_row = row + kh;
                    int input_col = col + kw;

                    if (input_row < input_height && input_col < input_width) {
                        __nv_bfloat16 inp_val = float_to_bfloat16(input_tensor[
                            batch * in_channels * input_height * input_width +
                            in_channel * input_height * input_width + input_row * input_width + input_col]);

                        __nv_bfloat16 weight_val = float_to_bfloat16(weight[
                            out_channel * in_channels * kernel_height * kernel_width +
                            in_channel * kernel_height * kernel_width + kh * kernel_width + kw]);

                        sum += bfloat16_to_float(__hmul(inp_val, weight_val));
                    }
                }
            }
        }
        sum += bfloat16_to_float(float_to_bfloat16(bias[out_channel]));
        output[batch * out_channels * input_height * input_width +
               out_channel * input_height * input_width + row * input_width + col] = fmaxf(sum, 0.0f); // ReLU
    }
}

extern "C" {

void torch_conv2d_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);  // batch_size
    int input_tensor_dim1 = va_arg(args, int);  // in_channels
    int input_tensor_dim2 = va_arg(args, int);  // input_height
    int input_tensor_dim3 = va_arg(args, int);  // input_width

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);  // out_channels
    int weight_dim1 = va_arg(args, int);  // in_channels
    int weight_dim2 = va_arg(args, int);  // kernel_height
    int weight_dim3 = va_arg(args, int);  // kernel_width

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);  // out_channels

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int input_height = input_tensor_dim2;
    int input_width = input_tensor_dim3;
    int out_channels = weight_dim0;
    int kernel_height = weight_dim2;
    int kernel_width = weight_dim3;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * in_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_weight, out_channels * in_channels * kernel_height * kernel_width * sizeof(float));
    cudaMalloc(&d_bias, out_channels * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * input_height * input_width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * in_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, out_channels * in_channels * kernel_height * kernel_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_height + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   out_channels, batch_size);

    conv2d_relu_kernel_bf16<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_bias, d_output, 
                                                            batch_size, in_channels, out_channels, 
                                                            input_height, input_width, kernel_height, kernel_width);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * out_channels * input_height * input_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
