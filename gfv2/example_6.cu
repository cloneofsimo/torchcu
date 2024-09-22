
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdarg.h>

#include "cutlass/cutlass.h"

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for affine grid generation using cutlass
__global__ void affine_grid_generator_kernel_fp16(const float* input_tensor, const float* theta, half* output,
                                                  int batch_size, int height, int width, int channels) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;

    if (batch_idx < batch_size && y < height && x < width) {
        int idx = batch_idx * height * width * channels + y * width * channels + x * channels;
        float grid_x = (x + 0.5f) / width * 2.0f - 1.0f;
        float grid_y = (y + 0.5f) / height * 2.0f - 1.0f;

        float tx = theta[batch_idx * 6 + 0] * grid_x + theta[batch_idx * 6 + 1] * grid_y + theta[batch_idx * 6 + 2];
        float ty = theta[batch_idx * 6 + 3] * grid_x + theta[batch_idx * 6 + 4] * grid_y + theta[batch_idx * 6 + 5];

        output[idx + 0] = float_to_half(tx);
        output[idx + 1] = float_to_half(ty);
    }
}

extern "C" {
    void torch_affine_grid_generator_fp16(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract input tensor
        const float* input_tensor = va_arg(args, const float*);
        int input_tensor_dim0 = va_arg(args, int);
        int input_tensor_dim1 = va_arg(args, int);
        int input_tensor_dim2 = va_arg(args, int);
        int input_tensor_dim3 = va_arg(args, int);

        // Extract theta tensor
        const float* theta = va_arg(args, const float*);
        int theta_dim0 = va_arg(args, int);
        int theta_dim1 = va_arg(args, int);
        int theta_dim2 = va_arg(args, int);

        // Extract output tensor (assuming it's preallocated)
        half* output = va_arg(args, half*);

        va_end(args);

        int batch_size = input_tensor_dim0;
        int height = input_tensor_dim2;
        int width = input_tensor_dim3;
        int channels = 2; // grid coordinates

        // Allocate device memory
        float *d_input, *d_theta;
        half *d_output;
        cudaMalloc(&d_input, batch_size * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float));
        cudaMalloc(&d_theta, theta_dim0 * theta_dim1 * theta_dim2 * sizeof(float));
        cudaMalloc(&d_output, batch_size * height * width * channels * sizeof(half));

        // Copy input data to device
        cudaMemcpy(d_input, input_tensor, batch_size * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_theta, theta, theta_dim0 * theta_dim1 * theta_dim2 * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (width + threadsPerBlock.z - 1) / threadsPerBlock.z);

        affine_grid_generator_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
            d_input, d_theta, d_output,
            batch_size, height, width, channels
        );

        // Copy result back to host
        cudaMemcpy(output, d_output, batch_size * height * width * channels * sizeof(half), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_theta);
        cudaFree(d_output);
    }
}
