
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/LinearCombination.h>
#include <cutlass/epilogue/threadblock/SaturatedRounding.h>
#include <cutlass/epilogue/threadblock/ThreadblockSaturatedRounding.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/Gemm.h>
#include <cutlass/layout/TensorNHWC.h>
#include <cutlass/layout/TensorNCHW.h>
#include <cutlass/matrix_traits.h>
#include <cutlass/numeric_types.h>
#include <cutlass/transform/threadblock/GemmTransform.h>
#include <cutlass/transform/threadblock/GemmTransformAdd.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/device/softmax.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#define CUDA_CHECK(ans)                                          \
  {                                                             \
    cudaError_t err = (ans);                                     \
    if (err != cudaSuccess) {                                    \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)     \
                << " in file " << __FILE__ << " at line " << __LINE__ \
                << std::endl;                                   \
      throw std::runtime_error("CUDA Error");                   \
    }                                                           \
  }

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
  return __int_as_float((int)h);
}

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
  return __float_as_int(f);
}

// CUDA kernel for SimCLR loss calculation using FP16
__global__ void simclr_loss_kernel(
    const half* z1_norm, const half* z2_norm, const float temperature,
    const bool symmetric, half* sim_matrix, half* loss, int batch_size) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < batch_size && col < batch_size) {
    float sum = 0.0f;
    for (int i = 0; i < 128; ++i) {
      sum += half_to_float(z1_norm[row * 128 + i]) *
             half_to_float(z2_norm[col * 128 + i]);
    }
    sim_matrix[row * batch_size + col] = float_to_half(sum);

    if (row != col) {
      // Compute loss for off-diagonal elements
      sim_matrix[row * batch_size + col] =
          float_to_half(exp(half_to_float(sim_matrix[row * batch_size + col]) /
                            temperature));
    }
  }

  if (row == col) {
    // Compute loss for diagonal elements (positive pairs)
    float positives = half_to_float(sim_matrix[row * batch_size + col]);
    float sum = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
      sum += exp(half_to_float(sim_matrix[row * batch_size + i]) /
                   temperature);
    }
    loss[row] = float_to_half(-log(positives / sum));
  }
}

// CUDA kernel for normalizing input tensors using FP16
__global__ void normalize_kernel(const half* input, half* output,
                                int batch_size, int embedding_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < batch_size * embedding_dim) {
    float sum = 0.0f;
    for (int j = 0; j < embedding_dim; ++j) {
      sum += half_to_float(input[i + j * batch_size]);
    }
    sum = sqrt(sum);
    for (int j = 0; j < embedding_dim; ++j) {
      output[i + j * batch_size] =
          float_to_half(half_to_float(input[i + j * batch_size]) / sum);
    }
  }
}

//  Define the type for the GEMM operation
using Element = cutlass::half_t;
using Layout = cutlass::layout::TensorNHWC;
using Arch = cutlass::arch::Sm75;

//  Define the GEMM operation with Cutlass library
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::gemm::GemmShape<128, 128, 128>, Element, Layout, Element, Layout,
    Element, Layout, cutlass::arch::Sm75,
    cutlass::gemm::GemmMode::kGemm,
    cutlass::gemm::GemmAlgorithm::kDefault,
    cutlass::epilogue::threadblock::ThreadblockSaturatedRounding<
        cutlass::epilogue::threadblock::LinearCombination<
            Element, Element, cutlass::epilogue::threadblock::SaturatedRounding<
                              cutlass::epilogue::threadblock::DefaultRounding,
                              Element>>,
        Element>,
    cutlass::transform::threadblock::GemmTransform<
        cutlass::transform::threadblock::DefaultGemmTransform,
        Element, Layout>,
    cutlass::transform::threadblock::GemmTransformAdd<
        cutlass::transform::threadblock::DefaultGemmTransformAdd,
        Element, Layout>>;

//  Define the GEMM operation with Cutlass library
using GemmFp32 = cutlass::gemm::device::Gemm<
    cutlass::gemm::GemmShape<128, 128, 128>, float, Layout, float, Layout,
    float, Layout, cutlass::arch::Sm75,
    cutlass::gemm::GemmMode::kGemm,
    cutlass::gemm::GemmAlgorithm::kDefault,
    cutlass::epilogue::threadblock::ThreadblockSaturatedRounding<
        cutlass::epilogue::threadblock::LinearCombination<
            float, float, cutlass::epilogue::threadblock::SaturatedRounding<
                              cutlass::epilogue::threadblock::DefaultRounding,
                              float>>,
        float>,
    cutlass::transform::threadblock::GemmTransform<
        cutlass::transform::threadblock::DefaultGemmTransform,
        float, Layout>,
    cutlass::transform::threadblock::GemmTransformAdd<
        cutlass::transform::threadblock::DefaultGemmTransformAdd,
        float, Layout>>;

//  Define the GEMM operation with Cutlass library
using GemmFp32ToFp16 = cutlass::gemm::device::Gemm<
    cutlass::gemm::GemmShape<128, 128, 128>, float, Layout, float, Layout,
    half, Layout, cutlass::arch::Sm75,
    cutlass::gemm::GemmMode::kGemm,
    cutlass::gemm::GemmAlgorithm::kDefault,
    cutlass::epilogue::threadblock::ThreadblockSaturatedRounding<
        cutlass::epilogue::threadblock::LinearCombination<
            float, float, cutlass::epilogue::threadblock::SaturatedRounding<
                              cutlass::epilogue::threadblock::DefaultRounding,
                              half>>,
        half>,
    cutlass::transform::threadblock::GemmTransform<
        cutlass::transform::threadblock::DefaultGemmTransform,
        float, Layout>,
    cutlass::transform::threadblock::GemmTransformAdd<
        cutlass::transform::threadblock::DefaultGemmTransformAdd,
        float, Layout>>;

extern "C" {

void torch_function(int num_args, ...) {
  va_list args;
  va_start(args, num_args);

  // Extract input tensors
  const float* z1 = va_arg(args, const float*);
  const float* z2 = va_arg(args, const float*);
  const float* temperature_ptr = va_arg(args, const float*);
  const bool* symmetric_ptr = va_arg(args, const bool*);
  float* output_ptr = va_arg(args, float*);

  va_end(args);

  //  Convert temperature to float
  float temperature = *temperature_ptr;
  bool symmetric = *symmetric_ptr;

  int batch_size = 128;
  int embedding_dim = 128;

  // Allocate device memory
  half* d_z1;
  half* d_z2;
  half* d_z1_norm;
  half* d_z2_norm;
  half* d_sim_matrix;
  half* d_loss;

  cudaMalloc(&d_z1, batch_size * embedding_dim * sizeof(half));
  cudaMalloc(&d_z2, batch_size * embedding_dim * sizeof(half));
  cudaMalloc(&d_z1_norm, batch_size * embedding_dim * sizeof(half));
  cudaMalloc(&d_z2_norm, batch_size * embedding_dim * sizeof(half));
  cudaMalloc(&d_sim_matrix, batch_size * batch_size * sizeof(half));
  cudaMalloc(&d_loss, batch_size * sizeof(half));

  // Copy input data to device
  cudaMemcpy(d_z1, z1, batch_size * embedding_dim * sizeof(half),
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_z2, z2, batch_size * embedding_dim * sizeof(half),
              cudaMemcpyHostToDevice);

  // Normalize z1 and z2
  normalize_kernel<<<(batch_size * embedding_dim + 1023) / 1024, 1024>>>(
      d_z1, d_z1_norm, batch_size, embedding_dim);
  normalize_kernel<<<(batch_size * embedding_dim + 1023) / 1024, 1024>>>(
      d_z2, d_z2_norm, batch_size, embedding_dim);

  //  Perform the GEMM operation to compute the similarity matrix
  //  using cutlass
  GemmFp32ToFp16 gemm;
  cutlass::MatrixCoord m = batch_size, n = batch_size, k = embedding_dim;
  gemm.initialize(m, n, k);

  //  Perform the GEMM operation with Cutlass library
  gemm.execute(
      reinterpret_cast<const Element*>(d_z1_norm),
      reinterpret_cast<const Element*>(d_z2_norm),
      reinterpret_cast<Element*>(d_sim_matrix));

  // Calculate SimCLR loss
  simclr_loss_kernel<<<(batch_size + 31) / 32, 32>>>(
      d_z1_norm, d_z2_norm, temperature, symmetric, d_sim_matrix, d_loss,
      batch_size);

  // Copy result back to host
  cudaMemcpy(output_ptr, d_loss, batch_size * sizeof(half),
              cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_z1);
  cudaFree(d_z2);
  cudaFree(d_z1_norm);
  cudaFree(d_z2_norm);
  cudaFree(d_sim_matrix);
  cudaFree(d_loss);
}

}  // extern "C"
