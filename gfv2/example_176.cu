
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void inplace_square_kernel(float* data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] *= data[idx];
  }
}

extern "C" {

void inplace_square_function(int num_args, ...) {
  va_list args;
  va_start(args, num_args);

  // Extract input tensor
  float* data = va_arg(args, float*);
  int dim0 = va_arg(args, int);
  int dim1 = va_arg(args, int);

  va_end(args);

  int size = dim0 * dim1;

  // Allocate device memory
  float *d_data;
  cudaMalloc(&d_data, size * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  int threadsPerBlock = 256;
  int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  inplace_square_kernel<<<numBlocks, threadsPerBlock>>>(d_data, size);

  // Copy result back to host (no need for this since it's in-place)
  // cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_data);
}

} // extern "C"
