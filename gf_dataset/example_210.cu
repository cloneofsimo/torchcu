
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

#define CHECK(x) do {                                                        \
    if ((x) != cudaSuccess) {                                              \
        fprintf(stderr, "Error at line %d : %s\n", __LINE__, cudaGetErrorString(x)); \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for sparse convolution with FFT
__global__ void sparse_fft_conv_kernel(const float* input, const float* weight, float* output,
                                       int batch_size, int in_channels, int out_channels,
                                       int in_height, int in_width, int kernel_height,
                                       int kernel_width, int stride_height, int stride_width,
                                       int padding_height, int padding_width,
                                       const bool* mask) {

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < batch_size && out_y < (in_height + 2 * padding_height - kernel_height) / stride_height + 1 &&
        out_x < (in_width + 2 * padding_width - kernel_width) / stride_width + 1) {

        float sum = 0.0f;

        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                for (int c = 0; c < in_channels; ++c) {
                    int in_y = out_y * stride_height + ky - padding_height;
                    int in_x = out_x * stride_width + kx - padding_width;

                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width &&
                        mask[c * kernel_height * kernel_width + ky * kernel_width + kx]) {
                        sum += input[b * in_channels * in_height * in_width + c * in_height * in_width +
                                    in_y * in_width + in_x] *
                               weight[c * out_channels * kernel_height * kernel_width +
                                      ky * kernel_width + kx];
                    }
                }
            }
        }
        output[b * out_channels * (in_height + 2 * padding_height - kernel_height) / stride_height + 1 *
                 (in_width + 2 * padding_width - kernel_width) / stride_width + 1 +
                 out_y * (in_width + 2 * padding_width - kernel_width) / stride_width + 1 + out_x] = sum;
    }
}

// CUDA kernel for adaptive max pooling
__global__ void adaptive_max_pool_kernel(const float* input, float* output, int batch_size, int in_channels,
                                         int in_height, int in_width, int out_height, int out_width) {

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int out_y = blockIdx.z * blockDim.z + threadIdx.z;
    int out_x = threadIdx.z;

    if (b < batch_size && c < in_channels && out_y < out_height && out_x < out_width) {
        int in_y_start = (out_y * in_height) / out_height;
        int in_y_end = ((out_y + 1) * in_height) / out_height;
        int in_x_start = (out_x * in_width) / out_width;
        int in_x_end = ((out_x + 1) * in_width) / out_width;

        float max_val = -INFINITY;
        for (int in_y = in_y_start; in_y < in_y_end; ++in_y) {
            for (int in_x = in_x_start; in_x < in_x_end; ++in_x) {
                max_val = fmaxf(max_val, input[b * in_channels * in_height * in_width +
                                                 c * in_height * in_width +
                                                 in_y * in_width + in_x]);
            }
        }
        output[b * in_channels * out_height * out_width + c * out_height * out_width +
               out_y * out_width + out_x] = max_val;
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
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int in_height = input_tensor_dim2;
    int in_width = input_tensor_dim3;

    int out_channels = 16;
    int kernel_height = 3;
    int kernel_width = 3;
    int stride_height = 1;
    int stride_width = 1;
    int padding_height = 1;
    int padding_width = 1;
    int out_height = (in_height + 2 * padding_height - kernel_height) / stride_height + 1;
    int out_width = (in_width + 2 * padding_width - kernel_width) / stride_width + 1;

    // Allocate device memory for input, output, and weight
    float *d_input, *d_output, *d_weight;
    CHECK(cudaMalloc(&d_input, batch_size * in_channels * in_height * in_width * sizeof(float)));
    CHECK(cudaMalloc(&d_output, batch_size * out_channels * out_height * out_width * sizeof(float)));
    CHECK(cudaMalloc(&d_weight, in_channels * out_channels * kernel_height * kernel_width * sizeof(float)));

    // Allocate device memory for mask
    bool *d_mask;
    CHECK(cudaMalloc(&d_mask, in_channels * kernel_height * kernel_width * sizeof(bool)));

    // Initialize weight tensor (you can modify this to load from a file or use a different initialization)
    float weight_data[in_channels * out_channels * kernel_height * kernel_width];
    for (int i = 0; i < in_channels * out_channels * kernel_height * kernel_width; ++i) {
        weight_data[i] = (float)rand() / RAND_MAX;
    }
    CHECK(cudaMemcpy(d_weight, weight_data, in_channels * out_channels * kernel_height * kernel_width * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize mask tensor (checkerboard pattern)
    bool mask_data[in_channels * kernel_height * kernel_width];
    for (int c = 0; c < in_channels; ++c) {
        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                if ((ky + kx) % 2 == 0) {
                    mask_data[c * kernel_height * kernel_width + ky * kernel_width + kx] = true;
                } else {
                    mask_data[c * kernel_height * kernel_width + ky * kernel_width + kx] = false;
                }
            }
        }
    }
    CHECK(cudaMemcpy(d_mask, mask_data, in_channels * kernel_height * kernel_width * sizeof(bool), cudaMemcpyHostToDevice));

    // Copy input data to device
    CHECK(cudaMemcpy(d_input, input_tensor, batch_size * in_channels * in_height * in_width * sizeof(float), cudaMemcpyHostToDevice));

    // Launch convolution kernel
    dim3 block_size(16, 16, 1);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x,
                   (out_height + block_size.y - 1) / block_size.y,
                   (out_width + block_size.z - 1) / block_size.z);

    sparse_fft_conv_kernel<<<grid_size, block_size>>>(
        d_input, d_weight, d_output, batch_size, in_channels, out_channels,
        in_height, in_width, kernel_height, kernel_width, stride_height, stride_width,
        padding_height, padding_width, d_mask
    );

    // Launch adaptive max pooling kernel
    dim3 pool_block_size(16, 16, 1);
    dim3 pool_grid_size((batch_size + pool_block_size.x - 1) / pool_block_size.x,
                        (out_channels + pool_block_size.y - 1) / pool_block_size.y,
                        (out_height + pool_block_size.z - 1) / pool_block_size.z);

    adaptive_max_pool_kernel<<<pool_grid_size, pool_block_size>>>(
        d_output, d_output, batch_size, out_channels, out_height, out_width,
        out_height / 2, out_width / 2
    );

    // Copy result back to host
    CHECK(cudaMemcpy(output, d_output, batch_size * out_channels * out_height / 2 * out_width / 2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_weight));
    CHECK(cudaFree(d_mask));
}

} // extern "C"

