
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for lightweight convolution using FP16
__global__ void lightweight_conv_fp16_kernel(const float* input_tensor, const float* weight, const float* bias,
                                               float* output, int batch_size, int input_channels, int output_channels,
                                               int input_height, int input_width, int kernel_size, int stride, int padding) {

    int batch_idx = blockIdx.z;
    int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    int output_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_y < input_height && output_x < input_width) {
        for (int output_c = 0; output_c < output_channels; ++output_c) {
            float sum = 0.0f;
            for (int input_c = 0; input_c < input_channels; ++input_c) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int input_y = output_y * stride - padding + ky;
                        int input_x = output_x * stride - padding + kx;

                        if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                            half input_val = float_to_half(input_tensor[((batch_idx * input_channels + input_c) * input_height + input_y) * input_width + input_x]);
                            half weight_val = float_to_half(weight[(output_c * input_channels + input_c) * kernel_size * kernel_size + (ky * kernel_size + kx)]);
                            sum += half_to_float(__hmul(input_val, weight_val));
                        }
                    }
                }
            }
            output[((batch_idx * output_channels + output_c) * input_height + output_y) * input_width + output_x] = sum + bias[output_c];
        }
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);
    int weight_dim3 = va_arg(args, int);

    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_channels = input_tensor_dim1;
    int output_channels = weight_dim0;
    int input_height = input_tensor_dim2;
    int input_width = input_tensor_dim3;
    int kernel_size = weight_dim2;
    int stride = 1;
    int padding = 1;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * input_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_weight, output_channels * input_channels * kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_bias, output_channels * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_channels * input_height * input_width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_channels * input_channels * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   batch_size);

    lightweight_conv_fp16_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output,
        batch_size, input_channels, output_channels,
        input_height, input_width, kernel_size, stride, padding
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_channels * input_height * input_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

} // extern "C"
