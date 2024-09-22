
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h> 

__device__ __forceinline__ float rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}

__device__ __forceinline__ float fmaxf(float x, float y) {
    return (x > y) ? x : y;
}

__device__ __forceinline__ float fminf(float x, float y) {
    return (x < y) ? x : y;
}

__device__ __forceinline__ int8_t float_to_int8(float f) {
    return __int_as_float(f);
}

__device__ __forceinline__ float int8_to_float(int8_t i) {
    return __float_as_int(i);
}

__global__ void spectral_rolloff_kernel(const float* input, const float* window, float* output, int batch_size, int time_steps) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < batch_size && col < time_steps) {
    float sum = 0.0f;
    for (int i = 0; i < time_steps; ++i) {
      sum += input[row * time_steps + i] * window[i];
    }
    output[row * time_steps + col] = sum;
  }
}

__global__ void fft_abs_squared_kernel(const float* input, float* output, int batch_size, int time_steps) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < batch_size && col < time_steps) {
    output[row * time_steps + col] = input[row * time_steps + col] * input[row * time_steps + col];
  }
}

__global__ void roll_kernel(float* input, int batch_size, int time_steps) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < batch_size && col < time_steps) {
    if (col == 0) {
      input[row * time_steps + col] = input[row * time_steps + time_steps - 1];
    } else {
      input[row * time_steps + col] = input[row * time_steps + col - 1];
    }
  }
}

__global__ void mean_kernel(const float* input, float* output, int batch_size, int time_steps) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < batch_size) {
    float sum = 0.0f;
    for (int i = 0; i < time_steps; ++i) {
      sum += input[row * time_steps + i];
    }
    output[row] = sum / time_steps;
  }
}

__global__ void avg_pool1d_kernel(const float* input, float* output, int batch_size, int input_size, int kernel_size, int stride) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < batch_size && col < (input_size - kernel_size + 1) / stride) {
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
      sum += input[row * input_size + col * stride + i];
    }
    output[row * (input_size - kernel_size + 1) / stride + col] = sum / kernel_size;
  }
}

__global__ void interpolate_kernel(const float* input, float* output, int batch_size, int input_size, int output_size) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < batch_size && col < output_size) {
    float ratio = (float)input_size / (float)output_size;
    int index = (int)(col * ratio);

    // Linear interpolation
    if (index < input_size - 1) {
      float alpha = col * ratio - index;
      output[row * output_size + col] = (1.0f - alpha) * input[row * input_size + index] + alpha * input[row * input_size + index + 1];
    } else {
      output[row * output_size + col] = input[row * input_size + input_size - 1];
    }
  }
}

__global__ void addcmul_kernel(const float* input1, const float* input2, const float* input3, float* output, int batch_size, int num_filters) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < batch_size && col < num_filters) {
    output[row * num_filters + col] = input1[row * num_filters + col] + input2[row * num_filters + col] * input3[row * num_filters + col];
  }
}

__global__ void normalize_kernel(float* input, float* output, int batch_size, int num_filters) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < batch_size && col < num_filters) {
    float sum_squared = 0.0f;
    for (int i = 0; i < num_filters; ++i) {
      sum_squared += input[row * num_filters + i] * input[row * num_filters + i];
    }
    output[row * num_filters + col] = input[row * num_filters + col] * rsqrtf(sum_squared);
  }
}

extern "C" {

void torch_function(int num_args, ...) {
  va_list args;
  va_start(args, num_args);

  // Extract input tensors
  const float* input_tensor = va_arg(args, const float*);
  int input_tensor_dim0 = va_arg(args, int);
  int input_tensor_dim1 = va_arg(args, int);

  const float* filter_bank = va_arg(args, const float*);
  int filter_bank_dim0 = va_arg(args, int);
  int filter_bank_dim1 = va_arg(args, int);

  const float* window = va_arg(args, const float*);
  int window_dim0 = va_arg(args, int);

  // Extract output tensor (assuming it's preallocated)
  float* output = va_arg(args, float*);

  va_end(args);

  // Allocate device memory
  float *d_input, *d_filter_bank, *d_window, *d_spectral_rolloff, *d_filtered_signal, *d_interpolated_signal, *d_features;
  cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
  cudaMalloc(&d_filter_bank, filter_bank_dim0 * filter_bank_dim1 * sizeof(float));
  cudaMalloc(&d_window, window_dim0 * sizeof(float));
  cudaMalloc(&d_spectral_rolloff, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
  cudaMalloc(&d_filtered_signal, input_tensor_dim0 * filter_bank_dim0 * sizeof(float));
  cudaMalloc(&d_interpolated_signal, input_tensor_dim0 * filter_bank_dim0 * sizeof(float));
  cudaMalloc(&d_features, input_tensor_dim0 * filter_bank_dim0 * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter_bank, filter_bank, filter_bank_dim0 * filter_bank_dim1 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_window, window, window_dim0 * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernels
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((input_tensor_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (input_tensor_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);
  spectral_rolloff_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_window, d_spectral_rolloff, input_tensor_dim0, input_tensor_dim1);
  
  numBlocks = ((input_tensor_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                (input_tensor_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);
  fft_abs_squared_kernel<<<numBlocks, threadsPerBlock>>>(d_spectral_rolloff, d_spectral_rolloff, input_tensor_dim0, input_tensor_dim1);

  numBlocks = ((input_tensor_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                (input_tensor_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);
  roll_kernel<<<numBlocks, threadsPerBlock>>>(d_spectral_rolloff, input_tensor_dim0, input_tensor_dim1);

  numBlocks = ((input_tensor_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                (input_tensor_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);
  mean_kernel<<<numBlocks, threadsPerBlock>>>(d_spectral_rolloff, d_spectral_rolloff, input_tensor_dim0, input_tensor_dim1);
  
  numBlocks = ((filter_bank_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                (input_tensor_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);
  avg_pool1d_kernel<<<numBlocks, threadsPerBlock>>>(d_spectral_rolloff, d_filtered_signal, input_tensor_dim0, filter_bank_dim1, 256, 1);

  numBlocks = ((filter_bank_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                (input_tensor_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);
  interpolate_kernel<<<numBlocks, threadsPerBlock>>>(d_filtered_signal, d_interpolated_signal, input_tensor_dim0, filter_bank_dim1, filter_bank_dim0);

  numBlocks = ((filter_bank_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                (input_tensor_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);
  addcmul_kernel<<<numBlocks, threadsPerBlock>>>(d_filtered_signal, d_interpolated_signal, d_filter_bank, d_features, input_tensor_dim0, filter_bank_dim0);

  numBlocks = ((filter_bank_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                (input_tensor_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);
  normalize_kernel<<<numBlocks, threadsPerBlock>>>(d_features, d_features, input_tensor_dim0, filter_bank_dim0);

  // Copy result back to host
  cudaMemcpy(output, d_features, input_tensor_dim0 * filter_bank_dim0 * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_filter_bank);
  cudaFree(d_window);
  cudaFree(d_spectral_rolloff);
  cudaFree(d_filtered_signal);
  cudaFree(d_interpolated_signal);
  cudaFree(d_features);
}

}
