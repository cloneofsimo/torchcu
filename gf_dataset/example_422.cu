
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#define CUDA_CHECK(condition)                                                                    \
    {                                                                                        \
        cudaError_t error = condition;                                                       \
        if (error != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error: %s in file %s at line %d\n", cudaGetErrorString(error), \
                    __FILE__, __LINE__);                                                      \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                    \
    }

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for windowed attention
__global__ void windowed_attention_kernel(const float* input, half* output, int B, int C, int H, int W, int window_size) {
    int b = blockIdx.x;
    int c = threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < B && c < C && h < H && w < W) {
        int window_h = h / window_size;
        int window_w = w / window_size;
        float sum = 0.0f;
        for (int i = window_h * window_size; i < (window_h + 1) * window_size; i++) {
            for (int j = window_w * window_size; j < (window_w + 1) * window_size; j++) {
                sum += input[b * C * H * W + c * H * W + i * W + j];
            }
        }
        sum /= window_size * window_size;
        output[b * C * H * W + c * H * W + h * W + w] = float_to_half(sum * input[b * C * H * W + c * H * W + h * W + w]);
    }
}

// CUDA kernel for Prewitt gradient calculation
__global__ void prewitt_gradient_kernel(const half* input, float* output_x, float* output_y,
                                       int B, int C, int H, int W) {
    int b = blockIdx.x;
    int c = threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < B && c < C && h < H && w < W) {
        // Prewitt kernel for x-gradient
        float gx = half_to_float(input[b * C * H * W + c * H * W + (h - 1) * W + (w - 1)]) +
                   half_to_float(input[b * C * H * W + c * H * W + (h - 1) * W + w]) +
                   half_to_float(input[b * C * H * W + c * H * W + (h - 1) * W + (w + 1)]) -
                   half_to_float(input[b * C * H * W + c * H * W + (h + 1) * W + (w - 1)]) -
                   half_to_float(input[b * C * H * W + c * H * W + (h + 1) * W + w]) -
                   half_to_float(input[b * C * H * W + c * H * W + (h + 1) * W + (w + 1)]);

        // Prewitt kernel for y-gradient
        float gy = half_to_float(input[b * C * H * W + c * H * W + (h - 1) * W + (w - 1)]) +
                   half_to_float(input[b * C * H * W + c * H * W + h * W + (w - 1)]) +
                   half_to_float(input[b * C * H * W + c * H * W + (h + 1) * W + (w - 1)]) -
                   half_to_float(input[b * C * H * W + c * H * W + (h - 1) * W + (w + 1)]) -
                   half_to_float(input[b * C * H * W + c * H * W + h * W + (w + 1)]) -
                   half_to_float(input[b * C * H * W + c * H * W + (h + 1) * W + (w + 1)]);

        output_x[b * C * H * W + c * H * W + h * W + w] = gx;
        output_y[b * C * H * W + c * H * W + h * W + w] = gy;
    }
}

extern "C" {
    // Function to execute the CUDA kernels
    void torch_function(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract input tensor
        const float* input = va_arg(args, const float*);
        int B = va_arg(args, int);
        int C = va_arg(args, int);
        int H = va_arg(args, int);
        int W = va_arg(args, int);

        // Extract window size
        int window_size = va_arg(args, int);

        // Extract output tensors
        float* output_attention = va_arg(args, float*);
        float* output_gradient_x = va_arg(args, float*);
        float* output_gradient_y = va_arg(args, float*);

        va_end(args);

        // Allocate device memory
        half* d_input;
        float* d_output_attention;
        float* d_output_gradient_x;
        float* d_output_gradient_y;

        CUDA_CHECK(cudaMalloc(&d_input, B * C * H * W * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_output_attention, B * C * H * W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_gradient_x, B * C * H * W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_gradient_y, B * C * H * W * sizeof(float)));

        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input, input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));

        // Launch windowed attention kernel
        dim3 threadsPerBlock(C, 8, 8);
        dim3 numBlocks(B, (H + threadsPerBlock.y - 1) / threadsPerBlock.y, (W + threadsPerBlock.z - 1) / threadsPerBlock.z);
        windowed_attention_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output_attention, B, C, H, W, window_size);

        // Launch Prewitt gradient kernel
        prewitt_gradient_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output_gradient_x, d_output_gradient_y, B, C, H, W);

        // Copy outputs back to host
        CUDA_CHECK(cudaMemcpy(output_attention, d_output_attention, B * C * H * W * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(output_gradient_x, d_output_gradient_x, B * C * H * W * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(output_gradient_y, d_output_gradient_y, B * C * H * W * sizeof(float), cudaMemcpyDeviceToHost));

        // Free device memory
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output_attention));
        CUDA_CHECK(cudaFree(d_output_gradient_x));
        CUDA_CHECK(cudaFree(d_output_gradient_y));
    }
}
