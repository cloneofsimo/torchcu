
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function for converting float to half
__device__ __forceinline__ half float_to_half(float f) {
  return __float2half_rn(f);
}

// Helper function for converting half to float
__device__ __forceinline__ float half_to_float(half h) {
  return __half2float(h);
}

extern "C" {

void torch_function(int num_args, ...) {
  va_list args;
  va_start(args, num_args);

  const float* input_tensor = va_arg(args, const float*);
  int input_tensor_dim0 = va_arg(args, int);
  int input_tensor_dim1 = va_arg(args, int);

  const float* scale = va_arg(args, const float*);
  const float* offset = va_arg(args, const float*);

  float* output = va_arg(args, float*);

  va_end(args);

  // Allocate device memory
  float *d_input, *d_output;
  half *d_scale, *d_offset;
  cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
  cudaMalloc(&d_output, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
  cudaMalloc(&d_scale, sizeof(half));
  cudaMalloc(&d_offset, sizeof(half));

  // Copy input data to device
  cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_offset, offset, sizeof(float), cudaMemcpyHostToDevice);

  // Apply scale and offset on the device
  for (int i = 0; i < input_tensor_dim0 * input_tensor_dim1; ++i) {
    d_input[i] = half_to_float(float_to_half(d_input[i]) * *d_scale + *d_offset);
  }

  // Apply ReLU on the device
  for (int i = 0; i < input_tensor_dim0 * input_tensor_dim1; ++i) {
    d_input[i] = fmaxf(d_input[i], 0.0f);
  }

  // Copy output data back to host
  cudaMemcpy(output, d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_scale);
  cudaFree(d_offset);
}

}  // extern "C"
