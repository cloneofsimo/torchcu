
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/util/tensor.h>

#include <cassert>
#include <cstdio>
#include <limits>

#define CHECK(condition)                                                     \
  {                                                                        \
    if (!(condition)) {                                                    \
      fprintf(stderr, "Error: " #condition " failed at %s:%d\n", __FILE__, \
              __LINE__);                                                  \
      exit(1);                                                            \
    }                                                                      \
  }

using namespace cutlass;

extern "C" {

// CUDA kernel for bilateral filter
__global__ void bilateral_filter_kernel(
    const half* input, const half* output, int batch_size, int height, int width, int kernel_size, float sigma_color, float sigma_spatial) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int center_x = x;
        int center_y = y;
        float sum = 0.0f;
        float weight_sum = 0.0f;

        for (int ky = -kernel_size / 2; ky <= kernel_size / 2; ky++) {
            for (int kx = -kernel_size / 2; kx <= kernel_size / 2; kx++) {
                int neighbor_x = center_x + kx;
                int neighbor_y = center_y + ky;

                if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                    int neighbor_idx = (center_y * width + center_x) * batch_size + neighbor_x * batch_size + neighbor_y;
                    int current_idx = (center_y * width + center_x) * batch_size;

                    float color_weight = exp(-(input[neighbor_idx] - input[current_idx]) * (input[neighbor_idx] - input[current_idx]) / (2 * sigma_color * sigma_color));
                    float spatial_weight = exp(-(kx * kx + ky * ky) / (2 * sigma_spatial * sigma_spatial));
                    float weight = color_weight * spatial_weight;

                    sum += __int2float_rn(input[neighbor_idx]) * weight;
                    weight_sum += weight;
                }
            }
        }

        output[y * width + x] = __float2half_rn(sum / weight_sum);
    }
}

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const half* input_tensor = va_arg(args, const half*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract kernel size
    int kernel_size = va_arg(args, int);

    // Extract sigma color
    float sigma_color = va_arg(args, float);

    // Extract sigma spatial
    float sigma_spatial = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    half* output = va_arg(args, half*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;

    // Allocate device memory
    half* d_input, *d_output;
    cudaMalloc(&d_input, batch_size * height * width * sizeof(half));
    cudaMalloc(&d_output, batch_size * height * width * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * height * width * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    bilateral_filter_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, height, width, kernel_size, sigma_color, sigma_spatial
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * height * width * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
