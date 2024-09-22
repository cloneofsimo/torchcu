
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/LinearCombination.h>
#include <cutlass/epilogue/threadblock/LinearCombination.h>
#include <cutlass/gemm/device/Gemm.h>
#include <cutlass/gemm/device/GemmForward.h>
#include <cutlass/gemm/device/Gemm.h>
#include <cutlass/gemm/device/GemmForward.h>
#include <cutlass/layout/Tensor.h>
#include <cutlass/layout/PitchLinear.h>
#include <cutlass/matrix_traits.h>
#include <cutlass/reduction/threadblock/ReduceSum.h>
#include <cutlass/reduction/threadblock/ReduceSum.h>
#include <cutlass/transform/threadblock/GemmInputTransform.h>
#include <cutlass/transform/threadblock/GemmInputTransform.h>
#include <cutlass/transform/threadblock/GemmOutputTransform.h>
#include <cutlass/transform/threadblock/GemmOutputTransform.h>
#include <cutlass/util/Index.h>
#include <cutlass/util/TypeTraits.h>
#include <cutlass/util/TensorView.h>
#include <cutlass/util/TensorView.h>
#include <cutlass/util/arch.h>
#include <cutlass/util/arch.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/gemm.h>
#include <cutlass/util/reference/gemm.h>
#include <cutlass/util/reference/tensor_op.h>
#include <cutlass/util/reference/tensor_op.h>
#include <cutlass/util/thread_block.h>
#include <cutlass/util/thread_block.h>
#include <cutlass/util/type_traits.h>
#include <cutlass/util/type_traits.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

// using namespace cutlass;
using cutlass::arch::Sm75;
using cutlass::layout::PitchLinear;
using cutlass::layout::RowMajor;
using cutlass::layout::ColumnMajor;
using cutlass::MatrixLayout;
using cutlass::MatrixTraits;
using cutlass::epilogue::threadblock::LinearCombination;
using cutlass::epilogue::threadblock::LinearCombination;
using cutlass::gemm::device::Gemm;
using cutlass::gemm::device::GemmForward;
using cutlass::gemm::device::Gemm;
using cutlass::gemm::device::GemmForward;
using cutlass::reduction::threadblock::ReduceSum;
using cutlass::reduction::threadblock::ReduceSum;
using cutlass::transform::threadblock::GemmInputTransform;
using cutlass::transform::threadblock::GemmInputTransform;
using cutlass::transform::threadblock::GemmOutputTransform;
using cutlass::transform::threadblock::GemmOutputTransform;
using cutlass::TensorView;
using cutlass::TensorView;
using cutlass::host_tensor;
using cutlass::util::Index;
using cutlass::util::reference::gemm;
using cutlass::util::reference::gemm;
using cutlass::util::reference::tensor_op;
using cutlass::util::reference::tensor_op;
using cutlass::util::thread_block;
using cutlass::util::type_traits;
using cutlass::util::arch::Sm75;

// helper function to convert int8 to float
__device__ __forceinline__ float int8_to_float(int8_t val) {
  return static_cast<float>(val) / 127.0f;
}

// helper function to convert float to int8
__device__ __forceinline__ int8_t float_to_int8(float val) {
  return static_cast<int8_t>(roundf(val * 127.0f));
}

__global__ void matmul_kernel(
    const int8_t* d_A,
    const int8_t* d_B,
    const float* d_bias,
    float* d_C,
    int m,
    int n,
    int k,
    float alpha) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      sum += int8_to_float(d_A[row * k + i]) * int8_to_float(d_B[col * k + i]);
    }
    d_C[row * n + col] = alpha * sum + d_bias[col];
  }
}

__global__ void
argmax_kernel(const float* d_data, int* d_result, int m, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    int index = row * n + col;
    float max_val = d_data[index];
    int max_idx = col;
    for (int i = 0; i < n; ++i) {
      if (d_data[row * n + i] > max_val) {
        max_val = d_data[row * n + i];
        max_idx = i;
      }
    }
    d_result[index] = max_idx;
  }
}

__global__ void
l1_loss_kernel(const float* d_data, const float* d_target, float* d_loss,
               int m, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    d_loss[row * n + col] = fabsf(d_data[row * n + col] -
                                 d_target[row * n + col]);
  }
}

__global__ void
isin_kernel(const int* d_data, const int* d_labels, float* d_result,
            int m, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < m) {
    for (int i = 0; i < n; ++i) {
      if (d_data[idx] == d_labels[i]) {
        d_result[idx] = 1.0f;
        return;
      }
    }
  }
}

extern "C" {

void torch_model_l1_isin_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    
    const int8_t* model_weight = va_arg(args, const int8_t*);
    int model_weight_dim0 = va_arg(args, int);
    int model_weight_dim1 = va_arg(args, int);
    const float* model_bias = va_arg(args, const float*);
    int model_bias_dim = va_arg(args, int);
    
    const int* labels = va_arg(args, const int*);
    int labels_dim = va_arg(args, int);

    float* loss = va_arg(args, float*);
    float* isin = va_arg(args, float*);

    va_end(args);

    // int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = model_weight_dim0;
    int batch_size = input_tensor_dim0;

    // Allocate device memory
    float *d_input, *d_output, *d_bias, *d_loss, *d_target;
    int8_t *d_model_weight;
    int *d_predicted_class, *d_labels_device;

    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));
    cudaMalloc(&d_loss, batch_size * sizeof(float));
    cudaMalloc(&d_target, batch_size * output_dim * sizeof(float));
    cudaMalloc(&d_bias, output_dim * sizeof(float));
    cudaMalloc(&d_model_weight, model_weight_dim0 * model_weight_dim1 * sizeof(int8_t));
    cudaMalloc(&d_predicted_class, batch_size * sizeof(int));
    cudaMalloc(&d_labels_device, labels_dim * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_weight, model_weight, model_weight_dim0 * model_weight_dim1 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, model_bias, output_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels_device, labels, labels_dim * sizeof(int), cudaMemcpyHostToDevice);

    // Perform matrix multiplication using CUTLASS
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_model_weight, d_bias, d_output, batch_size, output_dim,
        input_dim, 1.0f);

    // Find argmax (predicted class)
    argmax_kernel<<<numBlocks, threadsPerBlock>>>(d_output,
                                           d_predicted_class, batch_size,
                                           output_dim);

    // One-hot encoding
    // cudaMemcpy(d_target, d_predicted_class, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute L1 loss
    l1_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_output, d_target, d_loss, batch_size, output_dim);

    // Check isin
    isin_kernel<<<(batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
             threadsPerBlock.x>>>(
        d_predicted_class, d_labels_device, isin, batch_size, labels_dim);

    // Copy results back to host
    cudaMemcpy(loss, d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(isin, isin, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bias);
    cudaFree(d_loss);
    cudaFree(d_target);
    cudaFree(d_model_weight);
    cudaFree(d_predicted_class);
    cudaFree(d_labels_device);
}

} // extern "C"

