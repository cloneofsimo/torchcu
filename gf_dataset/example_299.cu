
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

#include "cutlass/cutlass.h"

#define CHECK(x)                                                                 \
  {                                                                          \
    cudaError_t error = x;                                                  \
    if (error != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));          \
      exit(1);                                                             \
    }                                                                          \
  }

// Helper function to convert float to __half
__device__ __forceinline__ __half float_to_half(float f) {
  return __float2half_rn(f);
}

// Helper function to convert __half to float
__device__ __forceinline__ float half_to_float(__half h) {
  return __half2float(h);
}

__global__ void
fused_glu_layernorm_grad_magnitude(const float* input, const float* weight,
                                  const float* bias, const float* gamma,
                                  const float* beta, float* output,
                                  float* grad_magnitude, int batch_size,
                                  int seq_len, int hidden_size) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int s = blockIdx.y * blockDim.y + threadIdx.y;

  if (b < batch_size && s < seq_len) {
    float sum = 0.0f;
    for (int h = 0; h < hidden_size; ++h) {
      sum += weight[h * hidden_size + h] * input[b * seq_len * hidden_size +
                                                   s * hidden_size + h];
    }

    float mean = sum / hidden_size;
    float variance = 0.0f;
    for (int h = 0; h < hidden_size; ++h) {
      variance +=
          (input[b * seq_len * hidden_size + s * hidden_size + h] - mean) *
          (input[b * seq_len * hidden_size + s * hidden_size + h] - mean);
    }
    variance /= hidden_size;
    variance += 1e-5f;

    float stddev = sqrtf(variance);
    for (int h = 0; h < hidden_size; ++h) {
      output[b * seq_len * hidden_size + s * hidden_size + h] =
          gamma[h] *
              (input[b * seq_len * hidden_size + s * hidden_size + h] - mean) /
              stddev +
          beta[h];
      output[b * seq_len * hidden_size + s * hidden_size + h] *=
          (1.0f / (1.0f + expf(-output[b * seq_len * hidden_size +
                                        s * hidden_size + h])));
      grad_magnitude[b * seq_len * hidden_size + s * hidden_size + h] =
          fabsf(output[b * seq_len * hidden_size + s * hidden_size + h]);
    }
  }
}

extern "C" {
void torch_function(int num_args, ...) {
  va_list args;
  va_start(args, num_args);

  // Extract input tensors
  const float* input = va_arg(args, const float*);
  int input_dim0 = va_arg(args, int);
  int input_dim1 = va_arg(args, int);
  int input_dim2 = va_arg(args, int);

  const float* weight = va_arg(args, const float*);
  int weight_dim0 = va_arg(args, int);
  int weight_dim1 = va_arg(args, int);

  const float* bias = va_arg(args, const float*);
  int bias_dim0 = va_arg(args, int);

  const float* gamma = va_arg(args, const float*);
  int gamma_dim0 = va_arg(args, int);

  const float* beta = va_arg(args, const float*);
  int beta_dim0 = va_arg(args, int);

  // Extract output tensors
  float* output = va_arg(args, float*);
  float* grad_magnitude = va_arg(args, float*);

  va_end(args);

  // Launch kernel
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks((input_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (input_dim1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  fused_glu_layernorm_grad_magnitude<<<numBlocks, threadsPerBlock>>>(
      input, weight, bias, gamma, beta, output, grad_magnitude, input_dim0,
      input_dim1, input_dim2);

  CHECK(cudaDeviceSynchronize());
}
}
