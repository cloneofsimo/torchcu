
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_math.h>
#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for fading-in with exponential function
__global__ void fading_in_kernel(const float* input, float* output, float alpha, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = (1 - expf(-alpha * i)) * input[i];
  }
}

// CUDA kernel for linear layer (using CUDNN)
__global__ void linear_layer_kernel(const float* input, const float* weight, float* output, int batch_size, int in_features,
                                      int out_features) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size) {
    float sum = 0.0f;
    for (int j = 0; j < in_features; ++j) {
      sum += input[i * in_features + j] * weight[j * out_features + threadIdx.y];
    }
    output[i * out_features + threadIdx.y] = fmaxf(sum, 0.0f); // ReLU activation
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

  const float* alpha = va_arg(args, const float*);

  // Extract output tensor (assuming it's preallocated)
  float* output = va_arg(args, float*);

  va_end(args);

  int batch_size = input_tensor_dim0;
  int input_dim = input_tensor_dim1;

  // Allocate device memory
  float *d_input, *d_weight, *d_output, *d_alpha;
  cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
  cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * sizeof(float));
  cudaMalloc(&d_output, batch_size * input_dim * sizeof(float));
  cudaMalloc(&d_alpha, sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_alpha, alpha, sizeof(float), cudaMemcpyHostToDevice);

  // CUDNN FFT
  cudnnHandle_t cudnnHandle;
  cudnnCreate(&cudnnHandle);

  cudnnTensorDescriptor_t inputDesc, outputDesc;
  cudnnCreateTensorDescriptor(&inputDesc);
  cudnnCreateTensorDescriptor(&outputDesc);

  cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
  int dimA[4] = {1, batch_size, 1, input_dim};
  cudnnSetTensorNdDescriptor(inputDesc, dataType, 4, dimA, nullptr);
  cudnnSetTensorNdDescriptor(outputDesc, dataType, 4, dimA, nullptr);

  cudnnPlan_t plan;
  cudnnCreatePlan(&plan);
  cudnnSetPlanForward(plan, inputDesc, outputDesc, CUDNN_FFT_FORWARD, dataType, CUDNN_FFT_DEFAULT, CUDNN_FFT_DEFAULT, 0);

  cudnnExecutePlan(plan, cudnnHandle, d_input, d_output);

  // CUDNN Linear Layer
  cudnnTensorDescriptor_t input_linear_desc, output_linear_desc, weight_linear_desc;
  cudnnCreateTensorDescriptor(&input_linear_desc);
  cudnnCreateTensorDescriptor(&output_linear_desc);
  cudnnCreateTensorDescriptor(&weight_linear_desc);

  int dimB[4] = {1, batch_size, 1, weight_dim1};
  int dimC[4] = {1, 1, 1, weight_dim0 * weight_dim1};
  cudnnSetTensorNdDescriptor(input_linear_desc, dataType, 4, dimB, nullptr);
  cudnnSetTensorNdDescriptor(output_linear_desc, dataType, 4, dimB, nullptr);
  cudnnSetTensorNdDescriptor(weight_linear_desc, dataType, 4, dimC, nullptr);

  cudnnFilterDescriptor_t filterDesc;
  cudnnCreateFilterDescriptor(&filterDesc);
  cudnnSetFilterNdDescriptor(filterDesc, dataType, CUDNN_TENSOR_NCHW, 4, dimC, nullptr);

  cudnnConvolutionDescriptor_t convDesc;
  cudnnCreateConvolutionDescriptor(&convDesc);

  cudnnSetConvolutionNdDescriptor(convDesc, 0, CUDNN_CROSS_CHANNEL_DIVISION, 0, 0, 1, 1, 1, 1, 1, 1, 1, CUDNN_DATA_FLOAT);
  cudnnConvolutionForward(cudnnHandle, 1.0f, input_linear_desc, d_output, filterDesc, d_weight, convDesc,
                          CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, nullptr, 0.0f, output_linear_desc, d_output);

  // Fading-in
  fading_in_kernel<<<(input_dim + 255) / 256, 256>>>(d_output, d_output, *d_alpha, input_dim);

  // Inverse CUDNN FFT
  cudnnSetPlanForward(plan, inputDesc, outputDesc, CUDNN_FFT_INVERSE, dataType, CUDNN_FFT_DEFAULT, CUDNN_FFT_DEFAULT, 0);
  cudnnExecutePlan(plan, cudnnHandle, d_output, d_output);

  // Copy result back to host
  cudaMemcpy(output, d_output, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

  // Free CUDNN resources
  cudnnDestroyPlan(plan);
  cudnnDestroyTensorDescriptor(inputDesc);
  cudnnDestroyTensorDescriptor(outputDesc);
  cudnnDestroyTensorDescriptor(input_linear_desc);
  cudnnDestroyTensorDescriptor(output_linear_desc);
  cudnnDestroyTensorDescriptor(weight_linear_desc);
  cudnnDestroyFilterDescriptor(filterDesc);
  cudnnDestroyConvolutionDescriptor(convDesc);
  cudnnDestroy(cudnnHandle);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
  cudaFree(d_alpha);
}

}  // extern "C"
