
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

#define BLOCK_SIZE 16

// Helper function for int8 convolution (using integer arithmetic)
__device__ int8_t int8_conv(const int8_t* input, const int8_t* weight, int kernel_size, int stride) {
  int sum = 0;
  for (int i = 0; i < kernel_size; i++) {
    sum += input[i] * weight[i];
  }
  return (int8_t) sum;
}

// CUDA kernel for the forward pass
__global__ void forward_kernel(const float* input, float* output, int batch_size, int in_channels, 
                               int height, int width, int out_channels, int kernel_size) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    for (int b = 0; b < batch_size; b++) {
      float sum = 0.0f;
      for (int c = 0; c < out_channels; c++) {
        for (int i = 0; i < kernel_size; i++) {
          for (int j = 0; j < kernel_size; j++) {
            int input_row = row + i - 1;
            int input_col = col + j - 1;
            if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
              sum += input[(b * in_channels * height * width) + (c * height * width) + (input_row * width) + input_col];
            }
          }
        }
        output[(b * out_channels * height * width) + (c * height * width) + (row * width) + col] = sum;
      }
    }
  }
}

// CUDA kernel for instance normalization
__global__ void instance_norm_kernel(float* data, int batch_size, int channels, int height, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    for (int b = 0; b < batch_size; b++) {
      float sum = 0.0f;
      for (int c = 0; c < channels; c++) {
        sum += data[(b * channels * height * width) + (c * height * width) + (row * width) + col];
      }
      float mean = sum / (channels * 1.0f);
      float var = 0.0f;
      for (int c = 0; c < channels; c++) {
        var += (data[(b * channels * height * width) + (c * height * width) + (row * width) + col] - mean) *
               (data[(b * channels * height * width) + (c * height * width) + (row * width) + col] - mean);
      }
      float std = sqrt(var / (channels * 1.0f));
      for (int c = 0; c < channels; c++) {
        data[(b * channels * height * width) + (c * height * width) + (row * width) + col] =
            (data[(b * channels * height * width) + (c * height * width) + (row * width) + col] - mean) / std;
      }
    }
  }
}

// CUDA kernel for square root
__global__ void sqrt_kernel(float* data, int batch_size, int channels, int height, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    for (int b = 0; b < batch_size; b++) {
      for (int c = 0; c < channels; c++) {
        data[(b * channels * height * width) + (c * height * width) + (row * width) + col] = 
            sqrtf(data[(b * channels * height * width) + (c * height * width) + (row * width) + col]);
      }
    }
  }
}

// CUDA kernel for median calculation
__global__ void median_kernel(float* data, float* median_data, int batch_size, int channels, int height, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    for (int b = 0; b < batch_size; b++) {
      float* channel_data = &data[(b * channels * height * width) + (row * width) + col];
      float median = channel_data[0];
      for (int c = 1; c < channels; c++) {
        if (channel_data[c] < median) {
          median = channel_data[c];
        }
      }
      median_data[(b * height * width) + (row * width) + col] = median;
    }
  }
}

extern "C" {
void my_function(int num_args, ...) {
  va_list args;
  va_start(args, num_args);

  // Extract input tensor
  const float* input = va_arg(args, const float*);
  int batch_size = va_arg(args, int);
  int in_channels = va_arg(args, int);
  int height = va_arg(args, int);
  int width = va_arg(args, int);

  // Extract output tensor (assuming it's preallocated)
  float* output = va_arg(args, float*);

  va_end(args);

  // Allocate device memory for intermediate results
  float* d_data;
  cudaMalloc(&d_data, batch_size * 8 * height * width * sizeof(float));

  // Allocate device memory for median calculation
  float* d_median_data;
  cudaMalloc(&d_median_data, batch_size * height * width * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_data, input, batch_size * in_channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

  // Convolution (Forward Pass)
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  forward_kernel<<<numBlocks, threadsPerBlock>>>(d_data, d_data, batch_size, in_channels, height, width, 8, 3);

  // Instance Normalization
  instance_norm_kernel<<<numBlocks, threadsPerBlock>>>(d_data, batch_size, 8, height, width);

  // Square Root
  sqrt_kernel<<<numBlocks, threadsPerBlock>>>(d_data, batch_size, 8, height, width);

  // Median Calculation
  median_kernel<<<numBlocks, threadsPerBlock>>>(d_data, d_median_data, batch_size, 8, height, width);

  // Copy result back to host
  cudaMemcpy(output, d_median_data, batch_size * height * width * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_data);
  cudaFree(d_median_data);
}
}
