
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for instance normalization (in-place)
__global__ void instance_norm_inplace_kernel(float* input, const float* mean, const float* std, 
                                           int N, int C, int H, int W, float eps) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N && c < C) {
        float sum = 0.0f;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                sum += input[(n * C + c) * H * W + h * W + w];
            }
        }
        float mean_val = sum / (float)(H * W);
        float std_val = 0.0f;

        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float diff = input[(n * C + c) * H * W + h * W + w] - mean_val;
                std_val += diff * diff;
            }
        }
        std_val = sqrtf(std_val / (float)(H * W));
        input[(n * C + c) * H * W + h * W + w] = (input[(n * C + c) * H * W + h * W + w] - mean_val) / (std_val + eps);
    }
}

extern "C" {

void instance_norm_inplace(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    float* input = va_arg(args, float*);
    int N = va_arg(args, int);
    int C = va_arg(args, int);
    int H = va_arg(args, int);
    int W = va_arg(args, int);

    // Extract eps
    float eps = va_arg(args, double);

    va_end(args);

    // Allocate device memory for mean and std
    float* d_mean;
    cudaMalloc(&d_mean, N * C * sizeof(float));
    float* d_std;
    cudaMalloc(&d_std, N * C * sizeof(float));

    // Launch kernel for instance normalization
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((C + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    instance_norm_inplace_kernel<<<numBlocks, threadsPerBlock>>>(input, d_mean, d_std, N, C, H, W, eps);

    // Free device memory
    cudaFree(d_mean);
    cudaFree(d_std);
}

} // extern "C"
