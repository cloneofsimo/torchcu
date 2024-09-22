
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for lightweight convolution
__global__ void lightweight_conv2d_kernel(const float* input, const float* weight, float* output,
                                         int batch_size, int in_channels, int out_channels,
                                         int input_height, int input_width, int kernel_size, int stride) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int out_channel = blockIdx.y * blockDim.y + threadIdx.y;
    int out_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_channel < out_channels && out_row < input_height / stride) {
        int out_col = 0;
        int in_row = out_row * stride;
        int in_col = out_col * stride;

        float sum = 0.0f;
        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            for (int k_row = 0; k_row < kernel_size; ++k_row) {
                for (int k_col = 0; k_col < kernel_size; ++k_col) {
                    int in_row_idx = in_row + k_row;
                    int in_col_idx = in_col + k_col;
                    if (in_row_idx >= 0 && in_row_idx < input_height &&
                        in_col_idx >= 0 && in_col_idx < input_width) {
                        sum += input[batch_idx * in_channels * input_height * input_width +
                                    in_channel * input_height * input_width +
                                    in_row_idx * input_width + in_col_idx] *
                               weight[out_channel * in_channels * kernel_size * kernel_size +
                                      in_channel * kernel_size * kernel_size +
                                      k_row * kernel_size + k_col];
                    }
                }
            }
        }
        output[batch_idx * out_channels * input_height / stride * input_width / stride +
              out_channel * input_height / stride * input_width / stride +
              out_row * input_width / stride + out_col] = sum;
    }
}

// CUDA kernel for bfloat16 relu
__global__ void bf16_relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        __nv_bfloat16 val = float_to_bfloat16(input[idx]);
        output[idx] = bfloat16_to_float(__hmax(val, float_to_bfloat16(0.0f)));
    }
}

// CUDA kernel for average pooling
__global__ void avg_pool2d_kernel(const float* input, float* output, int batch_size, int channels,
                                  int input_height, int input_width, int kernel_size, int stride) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int channel = blockIdx.y * blockDim.y + threadIdx.y;
    int out_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && channel < channels && out_row < input_height / stride) {
        int out_col = 0;
        int in_row = out_row * stride;
        int in_col = out_col * stride;

        float sum = 0.0f;
        for (int k_row = 0; k_row < kernel_size; ++k_row) {
            for (int k_col = 0; k_col < kernel_size; ++k_col) {
                int in_row_idx = in_row + k_row;
                int in_col_idx = in_col + k_col;
                if (in_row_idx >= 0 && in_row_idx < input_height &&
                    in_col_idx >= 0 && in_col_idx < input_width) {
                    sum += input[batch_idx * channels * input_height * input_width +
                                channel * input_height * input_width +
                                in_row_idx * input_width + in_col_idx];
                }
            }
        }
        output[batch_idx * channels * input_height / stride * input_width / stride +
              channel * input_height / stride * input_width / stride +
              out_row * input_width / stride + out_col] = sum / (kernel_size * kernel_size);
    }
}

extern "C" {

void my_complex_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract weights
    const float* weight_conv = va_arg(args, const float*);
    int weight_conv_dim0 = va_arg(args, int);
    int weight_conv_dim1 = va_arg(args, int);
    int weight_conv_dim2 = va_arg(args, int);
    int weight_conv_dim3 = va_arg(args, int);

    // Extract output tensors (assuming they are preallocated)
    float* output_final = va_arg(args, float*);
    float* output_conv = va_arg(args, float*);
    float* output_identity = va_arg(args, float*);
    float* output_bf16 = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int input_height = input_tensor_dim2;
    int input_width = input_tensor_dim3;
    int out_channels = weight_conv_dim0;
    int kernel_size = weight_conv_dim2;
    int stride = 1; // Assuming stride of 1

    // Allocate device memory
    float *d_input, *d_weight_conv, *d_output_conv, *d_output_bf16, *d_output_final;
    cudaMalloc(&d_input, batch_size * in_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_weight_conv, out_channels * in_channels * kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_output_conv, batch_size * out_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_output_bf16, batch_size * out_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_output_final, batch_size * out_channels * input_height / 2 * input_width / 2 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * in_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_conv, weight_conv, out_channels * in_channels * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch lightweight convolution kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((input_height / stride + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (out_channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);
    lightweight_conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weight_conv, d_output_conv,
                                                     batch_size, in_channels, out_channels,
                                                     input_height, input_width, kernel_size, stride);

    // Copy output_conv back to host (for identity output)
    cudaMemcpy(output_conv, d_output_conv, batch_size * out_channels * input_height * input_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Launch bf16 relu kernel
    int bf16_size = batch_size * out_channels * input_height * input_width;
    bf16_relu_kernel<<<(bf16_size + 255) / 256, 256>>>(d_output_conv, d_output_bf16, bf16_size);

    // Copy output_bf16 back to host
    cudaMemcpy(output_bf16, d_output_bf16, batch_size * out_channels * input_height * input_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Launch average pooling kernel
    int output_final_height = input_height / 2;
    int output_final_width = input_width / 2;
    dim3 numBlocks_pool((output_final_height + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (out_channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                        (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);
    avg_pool2d_kernel<<<numBlocks_pool, threadsPerBlock>>>(d_output_bf16, d_output_final, batch_size, out_channels,
                                                       input_height, input_width, 2, 2);

    // Copy output_final back to host
    cudaMemcpy(output_final, d_output_final, batch_size * out_channels * output_final_height * output_final_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight_conv);
    cudaFree(d_output_conv);
    cudaFree(d_output_bf16);
    cudaFree(d_output_final);
}

}  // extern "C"
