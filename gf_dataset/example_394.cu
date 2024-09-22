
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel.h>
#include <cutlass/conv/conv2d.h>
#include <cutlass/conv/epilogue/threadblock/linear_combine.h>
#include <cutlass/conv/epilogue/threadblock/linear_combine_complex.h>
#include <cutlass/conv/epilogue/threadblock/scale_by_half.h>
#include <cutlass/conv/epilogue/threadblock/scale_by_half_complex.h>
#include <cutlass/conv/epilogue/threadblock/scale_by_one.h>
#include <cutlass/conv/epilogue/threadblock/scale_by_one_complex.h>
#include <cutlass/conv/epilogue/threadblock/tensor_op_combine.h>
#include <cutlass/conv/epilogue/threadblock/tensor_op_combine_complex.h>
#include <cutlass/conv/epilogue/threadblock/tensor_op_combine_identity.h>
#include <cutlass/conv/epilogue/threadblock/tensor_op_combine_identity_complex.h>
#include <cutlass/conv/epilogue/threadblock/tensor_op_combine_scalar.h>
#include <cutlass/conv/epilogue/threadblock/tensor_op_combine_scalar_complex.h>
#include <cutlass/conv/epilogue/threadblock/tensor_op_combine_vector.h>
#include <cutlass/conv/epilogue/threadblock/tensor_op_combine_vector_complex.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/gemm.h>
#include <cutlass/gemm/threadblock/mma_tensor_op.h>
#include <cutlass/gemm/threadblock/mma_tensor_op_complex.h>
#include <cutlass/gemm/threadblock/mma_tensor_op_scalar.h>
#include <cutlass/gemm/threadblock/mma_tensor_op_scalar_complex.h>
#include <cutlass/gemm/threadblock/mma_tensor_op_vector.h>
#include <cutlass/gemm/threadblock/mma_tensor_op_vector_complex.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_multiply/kernel.h>
#include <cutlass/matrix_multiply/matrix_multiply.h>
#include <cutlass/matrix_multiply/threadblock/mma_tensor_op.h>
#include <cutlass/matrix_multiply/threadblock/mma_tensor_op_complex.h>
#include <cutlass/matrix_multiply/threadblock/mma_tensor_op_scalar.h>
#include <cutlass/matrix_multiply/threadblock/mma_tensor_op_scalar_complex.h>
#include <cutlass/matrix_multiply/threadblock/mma_tensor_op_vector.h>
#include <cutlass/matrix_multiply/threadblock/mma_tensor_op_vector_complex.h>
#include <cutlass/reduction/device/reduction.h>
#include <cutlass/reduction/reduction.h>
#include <cutlass/reduction/threadblock/reduction.h>
#include <cutlass/reduction/threadblock/reduction_complex.h>
#include <cutlass/tensor_op/device/tensor_op.h>
#include <cutlass/tensor_op/tensor_op.h>
#include <cutlass/tensor_op/threadblock/tensor_op.h>
#include <cutlass/tensor_op/threadblock/tensor_op_complex.h>
#include <cutlass/transform/device/transform.h>
#include <cutlass/transform/transform.h>
#include <cutlass/transform/threadblock/transform.h>
#include <cutlass/transform/threadblock/transform_complex.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T>
__device__ __forceinline__ T abs(T x) {
  return (x < 0) ? -x : x;
}

template <typename T>
__device__ __forceinline__ T float_to_half(T x) {
  return __float2half(x);
}

template <typename T>
__device__ __forceinline__ T half_to_float(T x) {
  return __half2float(x);
}

__global__ void zero_crossing_rate_kernel(const float* input, const float* output, const int n_samples, const int n_channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_channels) {
    float count = 0.0f;
    for (int i = 1; i < n_samples; i++) {
      if (abs(input[idx * n_samples + i] - input[idx * n_samples + i - 1]) > 0.0f) {
        count += 1.0f;
      }
    }
    output[idx] = count / (n_samples - 1);
  }
}

__global__ void weight_sparsity_kernel(const float* weight, const float* output, const int n_rows, const int n_cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_rows * n_cols) {
    if (weight[idx] == 0.0f) {
      atomicAdd(output, 1.0f);
    }
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

  const float* weight = va_arg(args, const float*);
  int weight_dim0 = va_arg(args, int);
  int weight_dim1 = va_arg(args, int);

  // Extract output tensors
  float* output_zero_crossing_rate = va_arg(args, float*);
  float* output_weight_sparsity = va_arg(args, float*);

  va_end(args);

  // Zero-crossing rate
  float* d_input;
  float* d_output_zero_crossing_rate;
  cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
  cudaMalloc(&d_output_zero_crossing_rate, input_tensor_dim0 * sizeof(float));

  cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyHostToDevice);

  zero_crossing_rate_kernel<<<(input_tensor_dim0 + 255) / 256, 256>>>(d_input, d_output_zero_crossing_rate, input_tensor_dim1, input_tensor_dim0);

  cudaMemcpy(output_zero_crossing_rate, d_output_zero_crossing_rate, input_tensor_dim0 * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output_zero_crossing_rate);

  // Weight sparsity
  float* d_weight;
  float* d_output_weight_sparsity;
  cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * sizeof(float));
  cudaMalloc(&d_output_weight_sparsity, sizeof(float));

  cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * sizeof(float), cudaMemcpyHostToDevice);

  weight_sparsity_kernel<<<(weight_dim0 * weight_dim1 + 255) / 256, 256>>>(d_weight, d_output_weight_sparsity, weight_dim0, weight_dim1);

  cudaMemcpy(output_weight_sparsity, d_output_weight_sparsity, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_weight);
  cudaFree(d_output_weight_sparsity);
}
}
