
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <stdarg.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/epilogue/threadblock/linear_combination.h>
#include <cutlass/epilogue/threadblock/warp_reduce.h>
#include <cutlass/epilogue/threadblock/mma_tensor_op.h>

// Helper functions for fp16 conversions
__device__ __forceinline__ half float_to_half(float f) {
  return __float2half(f);
}

__device__ __forceinline__ float half_to_float(half h) {
  return __half2float(h);
}

// CUDA kernel for the relative positional encoding with median aggregation
template <typename T>
__global__ void rel_pos_median_kernel(const T* input_tensor, const T* rel_pos_emb,
                                      T* output, int batch_size, int seq_len, int head_num,
                                      int embed_dim_per_head) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < batch_size && col < seq_len) {
    int offset = row * seq_len * head_num * embed_dim_per_head + col * head_num * embed_dim_per_head;
    T sum = 0;
    for (int head = 0; head < head_num; ++head) {
      for (int i = 0; i < seq_len; ++i) {
        T val = input_tensor[offset + head * embed_dim_per_head + i * head_num * embed_dim_per_head] *
                rel_pos_emb[col * seq_len * head_num * embed_dim_per_head + i * head_num * embed_dim_per_head +
                          head * embed_dim_per_head];
        sum = fmaxf(sum, val); // Median aggregation
      }
    }
    output[row * seq_len * embed_dim_per_head * head_num + col * head_num * embed_dim_per_head] = sum;
  }
}

extern "C" {

void torch_function(int num_args, ...) {
  va_list args;
  va_start(args, num_args);

  // Extract arguments
  const float* input_tensor = va_arg(args, const float*);
  int input_tensor_dim0 = va_arg(args, int);
  int input_tensor_dim1 = va_arg(args, int);
  int input_tensor_dim2 = va_arg(args, int);
  const float* rel_pos_emb = va_arg(args, const float*);
  int rel_pos_emb_dim0 = va_arg(args, int);
  int rel_pos_emb_dim1 = va_arg(args, int);
  int rel_pos_emb_dim2 = va_arg(args, int);
  int head_num = va_arg(args, int);
  float* output = va_arg(args, float*);

  va_end(args);

  int batch_size = input_tensor_dim0;
  int seq_len = input_tensor_dim1;
  int embed_dim = input_tensor_dim2;
  int embed_dim_per_head = embed_dim / head_num;

  // Allocate device memory
  half *d_input_fp16, *d_rel_pos_emb_fp16, *d_output_fp16;
  cudaMalloc(&d_input_fp16, batch_size * seq_len * embed_dim * sizeof(half));
  cudaMalloc(&d_rel_pos_emb_fp16, seq_len * seq_len * head_num * embed_dim_per_head * sizeof(half));
  cudaMalloc(&d_output_fp16, batch_size * seq_len * head_num * embed_dim_per_head * sizeof(half));

  // Copy input data to device in fp16
  cudaMemcpy(d_input_fp16, input_tensor, batch_size * seq_len * embed_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_rel_pos_emb_fp16, rel_pos_emb, seq_len * seq_len * head_num * embed_dim_per_head * sizeof(float),
             cudaMemcpyHostToDevice);

  // Launch kernel
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

  rel_pos_median_kernel<<<numBlocks, threadsPerBlock>>>(d_input_fp16, d_rel_pos_emb_fp16, d_output_fp16,
                                                        batch_size, seq_len, head_num, embed_dim_per_head);

  // Copy result back to host in fp32
  cudaMemcpy(output, d_output_fp16, batch_size * seq_len * head_num * embed_dim_per_head * sizeof(half),
             cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input_fp16);
  cudaFree(d_rel_pos_emb_fp16);
  cudaFree(d_output_fp16);
}

}  // extern "C"
