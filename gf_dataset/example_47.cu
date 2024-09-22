
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Helper function for converting float to half
__device__ __forceinline__ half float_to_half(float f) {
  return __float2half_rn(f);
}

// Helper function for converting half to float
__device__ __forceinline__ float half_to_float(half h) {
  return __half2float(h);
}

// Kernel for convolution with ReLU activation
__global__ void conv2d_relu_kernel(const half* input, const half* weight, const half* bias,
                                   half* output, int batch_size, int in_channels, int out_channels,
                                   int input_height, int input_width, int kernel_size, int padding) {
  int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
  int out_channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int out_row = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch_idx < batch_size && out_channel_idx < out_channels && out_row < input_height) {
    int out_col = 0;
    float sum = 0.0f;

    for (int in_channel_idx = 0; in_channel_idx < in_channels; in_channel_idx++) {
      for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
        for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
          int input_row = out_row + kernel_row - padding;
          int input_col = out_col + kernel_col - padding;

          if (input_row >= 0 && input_row < input_height && input_col >= 0 && input_col < input_width) {
            sum += half_to_float(input[batch_idx * in_channels * input_height * input_width +
                                       in_channel_idx * input_height * input_width +
                                       input_row * input_width + input_col]) *
                   half_to_float(weight[out_channel_idx * in_channels * kernel_size * kernel_size +
                                        in_channel_idx * kernel_size * kernel_size +
                                        kernel_row * kernel_size + kernel_col]);
          }
        }
      }
    }

    sum += half_to_float(bias[out_channel_idx]);
    output[batch_idx * out_channels * input_height * input_width +
           out_channel_idx * input_height * input_width +
           out_row * input_width + out_col] = float_to_half(fmaxf(sum, 0.0f));
  }
}

// Kernel for adaptive average pooling
__global__ void adaptive_avg_pool2d_kernel(const half* input, half* output, int batch_size,
                                            int in_channels, int input_height, int input_width,
                                            int output_height, int output_width) {
  int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
  int out_channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int out_row = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch_idx < batch_size && out_channel_idx < in_channels && out_row < output_height) {
    int out_col = 0;
    float sum = 0.0f;

    int input_row_start = out_row * input_height / output_height;
    int input_row_end = (out_row + 1) * input_height / output_height;
    int input_col_start = out_col * input_width / output_width;
    int input_col_end = (out_col + 1) * input_width / output_width;

    for (int input_row = input_row_start; input_row < input_row_end; input_row++) {
      for (int input_col = input_col_start; input_col < input_col_end; input_col++) {
        sum += half_to_float(input[batch_idx * in_channels * input_height * input_width +
                                   out_channel_idx * input_height * input_width +
                                   input_row * input_width + input_col]);
      }
    }

    output[batch_idx * in_channels * output_height * output_width +
           out_channel_idx * output_height * output_width +
           out_row * output_width + out_col] = float_to_half(sum / (input_row_end - input_row_start) / (input_col_end - input_col_start));
  }
}

extern "C" {
void torch_function(int num_args, ...) {
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

  // Extract output tensor (assuming it's preallocated)
  float* output = va_arg(args, float*);

  va_end(args);

  int batch_size = input_tensor_dim0;
  int in_channels = input_tensor_dim1;
  int input_height = input_tensor_dim2;
  int input_width = input_tensor_dim3;
  int out_channels = weight_dim0;
  int kernel_size = weight_dim2;
  int padding = 1;
  int output_height = 1;
  int output_width = 1;

  // Allocate device memory
  half *d_input, *d_weight, *d_bias, *d_output;
  cudaMalloc(&d_input, batch_size * in_channels * input_height * input_width * sizeof(half));
  cudaMalloc(&d_weight, out_channels * in_channels * kernel_size * kernel_size * sizeof(half));
  cudaMalloc(&d_bias, out_channels * sizeof(half));
  cudaMalloc(&d_output, batch_size * out_channels * output_height * output_width * sizeof(half));

  // Copy input data to device
  cudaMemcpy(d_input, input_tensor, batch_size * in_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, weight, out_channels * in_channels * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);

  // First convolution with ReLU
  dim3 threadsPerBlock(16, 16, 1);
  dim3 numBlocks((input_height + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (out_channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);
  conv2d_relu_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_bias, d_output, batch_size,
                                                    in_channels, out_channels, input_height, input_width,
                                                    kernel_size, padding);

  // Second convolution with ReLU
  conv2d_relu_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_weight, d_bias, d_output, batch_size,
                                                    out_channels, out_channels, input_height, input_width,
                                                    kernel_size, padding);

  // Adaptive average pooling
  dim3 threadsPerBlock_pool(16, 16, 1);
  dim3 numBlocks_pool((output_height + threadsPerBlock_pool.x - 1) / threadsPerBlock_pool.x,
                       (out_channels + threadsPerBlock_pool.y - 1) / threadsPerBlock_pool.y,
                       (batch_size + threadsPerBlock_pool.z - 1) / threadsPerBlock_pool.z);
  adaptive_avg_pool2d_kernel<<<numBlocks_pool, threadsPerBlock_pool>>>(d_output, d_output, batch_size,
                                                                        out_channels, input_height, input_width,
                                                                        output_height, output_width);

  // Copy result back to host
  cudaMemcpy(output, d_output, batch_size * out_channels * output_height * output_width * sizeof(half), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_bias);
  cudaFree(d_output);
}
}
