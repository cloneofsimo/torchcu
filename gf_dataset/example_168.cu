
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/epilogue/threadblock/epilogue.h>
#include <cutlass/epilogue/threadblock/linear_combine.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/tensor_view.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host.h>
#include <cutlass/util/reference/device.h>
#include <cutlass/util/reference/gemm.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/matrix_multiply.h>
#include <cutlass/transform/threadblock/threadblock_transform.h>
#include <cutlass/arch/mma.h>

#include <stdarg.h>

// Helper function to convert float to __half
__device__ __forceinline__ __half float_to_half(float f) {
  return __float2half(f);
}

// Helper function to convert __half to float
__device__ __forceinline__ float half_to_float(__half h) {
  return __half2float(h);
}

// CUDA kernel for matrix multiplication and PReLU using FP32
template <typename T>
__global__ void feature_mixing_prelu_kernel(const T* input_tensor, const T* weight, const T* bias, const T* slope, T* output, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    T sum = T(0.0);
    for (int i = 0; i < k; ++i) {
      sum += input_tensor[row * k + i] * weight[col * k + i];
    }
    sum += bias[col];
    output[row * n + col] = (sum >= T(0.0)) ? sum : (sum * slope[col]);
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

  // Extract weight tensor
  const float* weight = va_arg(args, const float*);
  int weight_dim0 = va_arg(args, int);
  int weight_dim1 = va_arg(args, int);

  // Extract bias tensor
  const float* bias = va_arg(args, const float*);
  int bias_dim0 = va_arg(args, int);

  // Extract slope tensor
  const float* slope = va_arg(args, const float*);
  int slope_dim0 = va_arg(args, int);

  // Extract output tensor (assuming it's preallocated)
  float* output = va_arg(args, float*);

  va_end(args);

  int batch_size = input_tensor_dim0;
  int input_dim = input_tensor_dim1;
  int output_dim = weight_dim0;

  // Allocate device memory
  float *d_input, *d_weight, *d_bias, *d_slope, *d_output;
  cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
  cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
  cudaMalloc(&d_bias, output_dim * sizeof(float));
  cudaMalloc(&d_slope, output_dim * sizeof(float));
  cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, output_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_slope, slope, output_dim * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

  feature_mixing_prelu_kernel<<<numBlocks, threadsPerBlock>>>(
      d_input, d_weight, d_bias, d_slope, d_output, batch_size, output_dim, input_dim
  );

  // Copy result back to host
  cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_bias);
  cudaFree(d_slope);
  cudaFree(d_output);
}

}  // extern "C"
