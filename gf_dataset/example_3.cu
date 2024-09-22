
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cufft.h>
#include <stdarg.h>

#define CHECK(x)                                      \
  do {                                                \
    cudaError_t err = (x);                           \
    if (err != cudaSuccess) {                         \
      fprintf(stderr, "ERROR: %s:%d,  %s\n", __FILE__, \
              __LINE__, cudaGetErrorString(err));     \
      exit(EXIT_FAILURE);                            \
    }                                                \
  } while (0)

// Helper functions for conversion between fp16 and float
__device__ __forceinline__ half float_to_half(float f) {
  return __float2half_rn(f);
}

__device__ __forceinline__ float half_to_float(half h) {
  return __half2float(h);
}

// Helper functions for conversion between bf16 and float
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for softmax calculation
__global__ void softmax_kernel(const float* input, float* output, float scale, int batch, int dim, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < batch && j < size) {
    float sum = 0.0f;
    for (int k = 0; k < dim; k++) {
      sum += expf(input[i * dim * size + k * size + j] * scale);
    }
    output[i * dim * size + j] = expf(input[i * dim * size + j] * scale) / sum;
  }
}

// CUDA kernel for inverse FFT
__global__ void ifft_kernel(const float* input, float* output, int batch, int dim, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < batch && j < size) {
    // Assuming input is complex with real and imaginary parts interleaved
    float real = input[2 * (i * dim * size + j)];
    float imag = input[2 * (i * dim * size + j) + 1];

    // Calculate inverse FFT
    cufftComplex z;
    z.x = real;
    z.y = imag;

    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_C2R, batch * dim);
    CHECK(cufftExecR2C(plan, &z, &z, size));
    CHECK(cufftDestroy(plan));

    output[i * dim * size + j] = z.x; // Store real part
  }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    float scale = va_arg(args, double); 

    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = input_tensor_dim2;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * output_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch softmax kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    softmax_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, scale, batch_size, input_dim, output_dim);

    // Launch ifft kernel
    ifft_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_input, batch_size, input_dim, output_dim);

    // Copy result back to host
    cudaMemcpy(output, d_input, batch_size * input_dim * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
