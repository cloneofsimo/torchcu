
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cutlass.h>
#include <stdarg.h>

#define CUDA_CHECK(condition)                                                                        \
  {                                                                                               \
    cudaError_t error = condition;                                                                \
    if (error != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA Error: %s in %s at line %d\n", cudaGetErrorString(error), __FILE__,     \
              __LINE__);                                                                         \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                             \
  }

// Helper function to calculate the mean along the last dimension
template <typename T>
__global__ void calculate_mean_kernel(const T* data, T* mean, int batch_size, int feature_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    T sum = 0.0f;
    for (int i = 0; i < feature_size; ++i) {
      sum += data[idx * feature_size + i];
    }
    mean[idx] = sum / feature_size;
  }
}

// Helper function to calculate the standard deviation along the last dimension
template <typename T>
__global__ void calculate_std_kernel(const T* data, const T* mean, T* std, int batch_size,
                                       int feature_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    T sum_sq = 0.0f;
    for (int i = 0; i < feature_size; ++i) {
      T diff = data[idx * feature_size + i] - mean[idx];
      sum_sq += diff * diff;
    }
    std[idx] = sqrt(sum_sq / feature_size);
  }
}

// Helper function to standardize the weight
template <typename T>
__global__ void standardize_weight_kernel(const T* weight, const T* mean, const T* std,
                                           T* standardized_weight, int batch_size,
                                           int feature_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size * feature_size) {
    int row = idx / feature_size;
    int col = idx % feature_size;
    standardized_weight[idx] =
        (weight[idx] - mean[row]) / (std[row] + 1e-6f);  // Add a small value to avoid division by zero
  }
}

// Helper function to perform outer product
template <typename T>
__global__ void outer_product_kernel(const T* input, const T* weight, T* output, int batch_size,
                                      int input_size, int weight_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size * input_size * weight_size) {
    int batch = idx / (input_size * weight_size);
    int row = (idx % (input_size * weight_size)) / weight_size;
    int col = (idx % (input_size * weight_size)) % weight_size;
    output[idx] = input[batch * input_size + row] * weight[col * weight_size + row];
  }
}

// Helper function to apply ReLU activation
template <typename T>
__global__ void relu_kernel(T* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    output[idx] = max(output[idx], (T)0);
  }
}

extern "C" {
void torch_function(int num_args, ...) {
  va_list args;
  va_start(args, num_args);

  // Extract input tensor
  const float* input_tensor = va_arg(args, const float*);
  int input_tensor_dim0 = va_arg(args, int);
  int input_tensor_dim1 = va_arg(args, int);

  // Extract weight tensor
  const float* weight = va_arg(args, const float*);
  int weight_dim0 = va_arg(args, int);
  int weight_dim1 = va_arg(args, int);

  // Extract output tensor (assuming it's preallocated)
  float* output = va_arg(args, float*);

  va_end(args);

  int batch_size = input_tensor_dim0;
  int input_size = input_tensor_dim1;
  int weight_size = weight_dim1;

  // Allocate device memory
  float* d_input;
  float* d_weight;
  float* d_mean;
  float* d_std;
  float* d_standardized_weight;
  float* d_outer_product;
  float* d_output;

  CUDA_CHECK(cudaMalloc(&d_input, batch_size * input_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mean, batch_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_std, batch_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_standardized_weight, weight_dim0 * weight_dim1 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_outer_product, batch_size * input_size * weight_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, batch_size * input_size * weight_size * sizeof(float)));

  // Copy input data to device
  CUDA_CHECK(
      cudaMemcpy(d_input, input_tensor, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * sizeof(float), cudaMemcpyHostToDevice));

  // Calculate mean along the last dimension
  calculate_mean_kernel<<<(batch_size + 255) / 256, 256>>>(d_weight, d_mean, batch_size, weight_size);

  // Calculate standard deviation along the last dimension
  calculate_std_kernel<<<(batch_size + 255) / 256, 256>>>(d_weight, d_mean, d_std, batch_size,
                                                    weight_size);

  // Standardize the weight
  standardize_weight_kernel<<<(batch_size * weight_size + 255) / 256, 256>>>(
      d_weight, d_mean, d_std, d_standardized_weight, batch_size, weight_size);

  // Perform outer product
  outer_product_kernel<<<(batch_size * input_size * weight_size + 255) / 256, 256>>>(
      d_input, d_standardized_weight, d_outer_product, batch_size, input_size, weight_size);

  // Apply ReLU activation
  relu_kernel<<<(batch_size * input_size * weight_size + 255) / 256, 256>>>(d_outer_product,
                                                                        batch_size * input_size *
                                                                            weight_size);

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(output, d_outer_product,
                  batch_size * input_size * weight_size * sizeof(float), cudaMemcpyDeviceToHost));

  // Free device memory
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_weight));
  CUDA_CHECK(cudaFree(d_mean));
  CUDA_CHECK(cudaFree(d_std));
  CUDA_CHECK(cudaFree(d_standardized_weight));
  CUDA_CHECK(cudaFree(d_outer_product));
  CUDA_CHECK(cudaFree(d_output));
}
}
