
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/linear_combination.h>
#include <cutlass/epilogue/threadblock/scale_linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_traits.h>
#include <cutlass/numeric_types.h>
#include <cutlass/reduction/threadblock/reduce_sum.h>
#include <cutlass/transform/threadblock/copy.h>
#include <cutlass/util/tensor_view.h>

#define CUDA_CHECK(condition)                                                  \
  {                                                                          \
    cudaError_t error = condition;                                          \
    if (error != cudaSuccess) {                                             \
      fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(error));                                   \
      exit(1);                                                            \
    }                                                                        \
  }

// Define data types and layouts
using ElementA = float;
using ElementB = float;
using ElementC = float;
using LayoutA = cutlass::layout::TensorNHWC;
using LayoutB = cutlass::layout::TensorNHWC;
using LayoutC = cutlass::layout::TensorNHWC;

// Define GEMM operation
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::gemm::GemmShape<8, 8, 8>,
    cutlass::layout::RowMajor, cutlass::layout::ColumnMajor,
    cutlass::layout::RowMajor,
    ElementA, ElementB, ElementC,
    cutlass::arch::Sm80,
    cutlass::epilogue::threadblock::LinearCombination,
    cutlass::reduction::threadblock::ReduceSum<
        ElementC, cutlass::arch::Sm80,
        cutlass::threadblock::GemmThreadblockShape<8, 8>>>;

// Define workspace size for GEMM
constexpr int kWorkspaceSize = Gemm::kWorkspaceSize;

// Kernel for pre-activation, sum, and fp16 conversion
__global__ void preActivationSumFp16(const float* input1, const float* input2,
                                     const float* input3, float* output,
                                     int batch_size, int height, int width) {
  // Calculate thread index
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int tidy = threadIdx.y + blockIdx.y * blockDim.y;

  // Calculate global index
  int index = tidx + tidy * blockDim.x + blockIdx.x * blockDim.x * gridDim.x;

  // Check if the thread is within the bounds of the input tensors
  if (index < batch_size * height * width) {
    // Calculate input tensor indices
    int input1_idx = index;
    int input2_idx = index;
    int input3_idx = index;

    // Load data from input tensors
    float a = input1[input1_idx];
    float b = input2[input2_idx];
    float c = input3[input3_idx];

    // Perform pre-activation
    float pre_activated = __fmaxf(a, 0.0f) + b;

    // Sum and convert to fp16
    float sum = pre_activated + c;
    __half output_fp16 = __float2half_rn(sum);

    // Store the result in the output tensor
    output[index] = __half2float(output_fp16);
  }
}

// Host function
extern "C" {
void torch_function(int num_args, ...) {
  va_list args;
  va_start(args, num_args);

  // Extract input tensors
  const float* input1 = va_arg(args, const float*);
  int input1_dim0 = va_arg(args, int);
  int input1_dim1 = va_arg(args, int);
  int input1_dim2 = va_arg(args, int);

  const float* input2 = va_arg(args, const float*);
  int input2_dim0 = va_arg(args, int);
  int input2_dim1 = va_arg(args, int);
  int input2_dim2 = va_arg(args, int);

  const float* input3 = va_arg(args, const float*);
  int input3_dim0 = va_arg(args, int);
  int input3_dim1 = va_arg(args, int);
  int input3_dim2 = va_arg(args, int);

  // Extract output tensor (assuming it's preallocated)
  float* output = va_arg(args, float*);

  va_end(args);

  // Allocate device memory
  float* d_input1;
  float* d_input2;
  float* d_input3;
  float* d_output;
  CUDA_CHECK(cudaMalloc(&d_input1, input1_dim0 * input1_dim1 * input1_dim2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_input2, input2_dim0 * input2_dim1 * input2_dim2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_input3, input3_dim0 * input3_dim1 * input3_dim2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, input1_dim0 * input1_dim1 * input1_dim2 * sizeof(float)));

  // Copy input data to device
  CUDA_CHECK(cudaMemcpy(d_input1, input1,
                         input1_dim0 * input1_dim1 * input1_dim2 * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_input2, input2,
                         input2_dim0 * input2_dim1 * input2_dim2 * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_input3, input3,
                         input3_dim0 * input3_dim1 * input3_dim2 * sizeof(float),
                         cudaMemcpyHostToDevice));

  // Launch kernel
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(
      (input1_dim0 * input1_dim1 * input1_dim2 + threadsPerBlock.x - 1) /
          threadsPerBlock.x,
      1);
  preActivationSumFp16<<<numBlocks, threadsPerBlock>>>(
      d_input1, d_input2, d_input3, d_output, input1_dim0, input1_dim1,
      input1_dim2);

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(output, d_output,
                         input1_dim0 * input1_dim1 * input1_dim2 * sizeof(float),
                         cudaMemcpyDeviceToHost));

  // Free device memory
  CUDA_CHECK(cudaFree(d_input1));
  CUDA_CHECK(cudaFree(d_input2));
  CUDA_CHECK(cudaFree(d_input3));
  CUDA_CHECK(cudaFree(d_output));
}
} // extern "C"
