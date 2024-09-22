
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

#define CUDA_CHECK(x)                                                                   \
  do                                                                                 \
  {                                                                                  \
    cudaError_t err = (x);                                                           \
    if (err != cudaSuccess)                                                          \
    {                                                                                  \
      fprintf(stderr, "Cuda error at line %d in file %s: %s\n", __LINE__, __FILE__,   \
              cudaGetErrorString(err));                                            \
      exit(EXIT_FAILURE);                                                           \
    }                                                                                  \
  } while (0)

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
  return __float2half_rn(f);
}

// Helper function to convert half to int8
__device__ __forceinline__ int8_t half_to_int8(half h) {
  return __int_as_int8(static_cast<int>(h));
}

// CUDA kernel for pixel shuffle, mixup, int8 conversion, and ReLU
__global__ void pixel_shuffle_mixup_int8_fp16_kernel(const float* input_tensor, 
                                                   const float* gt, 
                                                   float alpha, 
                                                   half* output, 
                                                   int batch_size, 
                                                   int channels, 
                                                   int in_height, 
                                                   int in_width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int out_height = in_height * 2;
  int out_width = in_width * 2;

  if (row < out_height && col < out_width && row < batch_size * channels * out_height && 
      col < batch_size * channels * out_width) {
    // Calculate indices for input and output
    int in_row = row / 2;
    int in_col = col / 2;
    int in_index = (in_row * in_width + in_col) * channels;

    // Mixup factor
    float lam = __ldg(alpha);

    // Calculate mixed value
    float mixed_value = lam * input_tensor[in_index] + (1 - lam) * gt[in_index];

    // Convert to int8 and apply ReLU
    int8_t int8_value = half_to_int8(float_to_half(mixed_value));
    int8_value = __int_as_int8(max(0, static_cast<int>(int8_value)));

    // Store in output (fp16)
    output[row * out_width + col] = __int_as_half(int8_value);
  }
}

extern "C" {

void pixel_shuffle_mixup_int8_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract gt tensor
    const float* gt = va_arg(args, const float*);
    int gt_dim0 = va_arg(args, int);
    int gt_dim1 = va_arg(args, int);
    int gt_dim2 = va_arg(args, int);
    int gt_dim3 = va_arg(args, int);

    // Extract alpha
    float alpha = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    half* output = va_arg(args, half*);

    va_end(args);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((input_tensor_dim3 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (input_tensor_dim2 * 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    pixel_shuffle_mixup_int8_fp16_kernel<<<numBlocks, threadsPerBlock>>>(
        input_tensor, 
        gt, 
        alpha, 
        output, 
        input_tensor_dim0, 
        input_tensor_dim1, 
        input_tensor_dim2, 
        input_tensor_dim3
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}
}  // extern "C"
