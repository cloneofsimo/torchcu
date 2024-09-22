
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
  return __float2half_rn(f);
}

// CUDA kernel for matrix inversion
__global__ void matrix_inversion_kernel(const float *input_tensor, float *output_tensor, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
      sum += input_tensor[row * N + i] * input_tensor[i * N + col];
    }
    output_tensor[row * N + col] = sum;
  }
}

// CUDA kernel for inner product and sigmoid activation
__global__ void inner_product_sigmoid_kernel(const float *input_tensor, half *output_tensor, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
      sum += input_tensor[row * N + i] * input_tensor[col * N + i];
    }
    output_tensor[row * N + col] = float_to_half(1.0f / (1.0f + expf(-sum)));
  }
}

extern "C" {

void my_function(int num_args, ...) {
  va_list args;
  va_start(args, num_args);

  // Extract input tensor
  const float* input_tensor = va_arg(args, const float*);
  int input_tensor_dim0 = va_arg(args, int);
  int input_tensor_dim1 = va_arg(args, int);

  // Extract output tensor (assuming it's preallocated)
  half* output = va_arg(args, half*);

  va_end(args);

  int N = input_tensor_dim0;

  // Allocate device memory
  float *d_input, *d_inverted;
  cudaMalloc(&d_input, N * N * sizeof(float));
  cudaMalloc(&d_inverted, N * N * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_input, input_tensor, N * N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch matrix inversion kernel
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
  matrix_inversion_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_inverted, N);

  // Launch inner product and sigmoid kernel
  half *d_output;
  cudaMalloc(&d_output, N * N * sizeof(half));
  inner_product_sigmoid_kernel<<<numBlocks, threadsPerBlock>>>(d_inverted, d_output, N);

  // Copy result back to host
  cudaMemcpy(output, d_output, N * N * sizeof(half), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_inverted);
  cudaFree(d_output);
}

}
