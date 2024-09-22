
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/linear_combination.h>
#include <cutlass/epilogue/threadblock/linear_combination_tensor_op.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_multiply/kernel.h>
#include <cutlass/matrix_multiply/threadblock/mma_tensor_op.h>
#include <cutlass/matrix_multiply/threadblock/mma_tensor_op_sm80.h>
#include <cutlass/matrix_multiply/threadblock/mma_tensor_op_sm86.h>
#include <cutlass/matrix_multiply/threadblock/mma_tensor_op_sm90.h>
#include <cutlass/matrix_multiply/threadblock/mma_tensor_op_sm92.h>
#include <cutlass/matrix_multiply/threadblock/mma_tensor_op_sm93.h>
#include <cutlass/reduction/threadblock/reduction.h>
#include <cutlass/reduction/threadblock/reduction_tensor_op.h>
#include <cutlass/transform/threadblock/predicated_tile_store.h>
#include <cutlass/transform/threadblock/predicated_tile_store_tensor_op.h>
#include <cutlass/util/arch.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device_tensor.h>
#include <cutlass/util/reference/matrix_multiply.h>
#include <cutlass/util/reference/reduction.h>
#include <cutlass/util/reference/tensor_op.h>
#include <cutlass/util/tensor_view.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#define CUTLASS_CHECK_CUDA_ERROR(status)                                     \
  if (status != cudaSuccess) {                                              \
    std::cerr << "Error: " << cudaGetErrorString(status) << " (" << status  \
              << ")" << std::endl;                                          \
  }

typedef cutlass::layout::TensorNHWC  layout_type;
typedef float                           element_type;

template <typename T>
__device__ __forceinline__ T min(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
__device__ __forceinline__ T max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__device__ __forceinline__ T clamp(T x, T minVal, T maxVal) {
  return min(max(x, minVal), maxVal);
}

template <typename T>
struct median {
  __device__ __forceinline__ T operator()(T *data, int count) {
    int middle = (count + 1) / 2;
    std::sort(data, data + count);
    return data[middle - 1];
  }
};

template <typename T>
struct ge {
  __device__ __forceinline__ bool operator()(T x, T y) {
    return x >= y;
  }
};

template <typename T>
struct ge_backward {
  __device__ __forceinline__ T operator()(T x, T y) {
    return x >= y ? T(1) : T(0);
  }
};

template <typename T>
struct ge_backward_grad_a {
  __device__ __forceinline__ T operator()(T x, T y) {
    return x >= y ? T(1) : T(0);
  }
};

template <typename T>
struct ge_backward_grad_b {
  __device__ __forceinline__ T operator()(T x, T y) {
    return T(0);
  }
};

extern "C" {

void torch_function(int num_args, ...) {
  va_list args;
  va_start(args, num_args);

  const float *input_tensor = va_arg(args, const float *);
  int input_tensor_dim0 = va_arg(args, int);
  int input_tensor_dim1 = va_arg(args, int);

  const float *threshold = va_arg(args, const float *);
  int threshold_dim0 = va_arg(args, int);

  float *output = va_arg(args, float *);

  va_end(args);

  // Create a Cutlass tensor descriptor for the input
  cutlass::TensorView<element_type, layout_type> input_tensor_view(
      {input_tensor_dim0, input_tensor_dim1}, {1, 1},
      layout_type::kStride, input_tensor);

  // Create a Cutlass tensor descriptor for the threshold
  cutlass::TensorView<element_type, layout_type> threshold_view(
      {threshold_dim0}, {1}, layout_type::kStride, threshold);

  // Create a Cutlass tensor descriptor for the output
  cutlass::TensorView<element_type, layout_type> output_view(
      {input_tensor_dim0, input_tensor_dim1}, {1, 1},
      layout_type::kStride, output);

  // Allocate device memory
  float *d_input, *d_threshold, *d_output;
  CUTLASS_CHECK_CUDA_ERROR(
      cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(float)));
  CUTLASS_CHECK_CUDA_ERROR(
      cudaMalloc(&d_threshold, threshold_dim0 * sizeof(float)));
  CUTLASS_CHECK_CUDA_ERROR(
      cudaMalloc(&d_output, input_tensor_dim0 * input_tensor_dim1 * sizeof(float)));

  // Copy input data to device
  CUTLASS_CHECK_CUDA_ERROR(cudaMemcpy(
      d_input, input_tensor,
      input_tensor_dim0 * input_tensor_dim1 * sizeof(float),
      cudaMemcpyHostToDevice));
  CUTLASS_CHECK_CUDA_ERROR(cudaMemcpy(
      d_threshold, threshold, threshold_dim0 * sizeof(float),
      cudaMemcpyHostToDevice));

  cutlass::TensorView<element_type, layout_type> d_input_view(
      {input_tensor_dim0, input_tensor_dim1}, {1, 1},
      layout_type::kStride, d_input);
  cutlass::TensorView<element_type, layout_type> d_threshold_view(
      {threshold_dim0}, {1}, layout_type::kStride, d_threshold);
  cutlass::TensorView<element_type, layout_type> d_output_view(
      {input_tensor_dim0, input_tensor_dim1}, {1, 1},
      layout_type::kStride, d_output);

  // Perform the median and ge operation on the device
  cutlass::Reduction::Threadblock<
      cutlass::arch::Sm80,
      cutlass::reduction::ThreadblockReduction::kReduceSum, element_type,
      element_type, cutlass::layout::RowMajor,
      cutlass::reduction::ThreadblockReduction::kDefaultEpilogue,
      median<element_type>, element_type, element_type,
      cutlass::layout::RowMajor>
      median_op;

  median_op.execute(d_input_view, d_output_view,
                    cutlass::TensorView<element_type, layout_type>{});

  cutlass::TensorOp::Threadblock<
      cutlass::arch::Sm80, element_type, element_type,
      cutlass::layout::RowMajor, cutlass::layout::RowMajor,
      cutlass::TensorOp::kElementwise, ge<element_type>,
      element_type, element_type, cutlass::layout::RowMajor>
      ge_op;

  ge_op.execute(d_output_view, d_threshold_view, d_output_view);

  CUTLASS_CHECK_CUDA_ERROR(cudaMemcpy(
      output, d_output, input_tensor_dim0 * input_tensor_dim1 * sizeof(float),
      cudaMemcpyDeviceToHost));

  // Free device memory
  CUTLASS_CHECK_CUDA_ERROR(cudaFree(d_input));
  CUTLASS_CHECK_CUDA_ERROR(cudaFree(d_threshold));
  CUTLASS_CHECK_CUDA_ERROR(cudaFree(d_output));
}

}
