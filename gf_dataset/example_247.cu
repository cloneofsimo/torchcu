
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

#define CUDA_CHECK(x) do {                                        \
    cudaError_t error = (x);                                     \
    if (error != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error: %s in %s at line %d\n",     \
                cudaGetErrorString(error),                        \
                __FILE__, __LINE__);                             \
        exit(1);                                                 \
    }                                                           \
} while(0)

extern "C" {

__global__ void bilateral_filter_threshold_kernel(
    const half* input_tensor, 
    const half* output_tensor, 
    const int* kernel_size,
    const float* sigma_color, 
    const float* sigma_space,
    const float* threshold, 
    const int height, 
    const int width,
    const int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width * channels + x * channels;

    // Bilateral Filtering
    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int dy = -(*kernel_size) / 2; dy <= (*kernel_size) / 2; ++dy) {
        for (int dx = -(*kernel_size) / 2; dx <= (*kernel_size) / 2; ++dx) {
            int neighbor_x = x + dx;
            int neighbor_y = y + dy;
            if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                int neighbor_idx = neighbor_y * width * channels + neighbor_x * channels;

                // Calculate spatial weight
                float spatial_weight = exp(-(dx * dx + dy * dy) / (2.0f * (*sigma_space) * (*sigma_space)));

                // Calculate color weight
                float color_weight = 1.0f; 
                for (int c = 0; c < channels; ++c) {
                    color_weight *= exp(-pow(input_tensor[neighbor_idx + c] - input_tensor[idx + c], 2.0f) / (2.0f * (*sigma_color) * (*sigma_color)));
                }

                // Weighted sum
                for (int c = 0; c < channels; ++c) {
                    sum += input_tensor[neighbor_idx + c] * spatial_weight * color_weight;
                }
                weight_sum += spatial_weight * color_weight;
            }
        }
    }

    // Normalize and clamp the filtered value
    for (int c = 0; c < channels; ++c) {
        output_tensor[idx + c] = __float2half_rn(sum / weight_sum);
    }
}

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    int kernel_size = va_arg(args, int);
    float sigma_color = va_arg(args, float);
    float sigma_space = va_arg(args, float);
    float threshold = va_arg(args, float);

    float* output_tensor = va_arg(args, float*);

    va_end(args);

    const int height = input_tensor_dim0;
    const int width = input_tensor_dim1;
    const int channels = input_tensor_dim2;

    // Allocate device memory for input and output tensors
    half* d_input;
    half* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, height * width * channels * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, height * width * channels * sizeof(half)));

    // Copy input tensor to device memory
    CUDA_CHECK(cudaMemcpy(d_input, input_tensor, height * width * channels * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate device memory for kernel size, sigma_color, sigma_space and threshold
    int* d_kernel_size;
    float* d_sigma_color;
    float* d_sigma_space;
    float* d_threshold;
    CUDA_CHECK(cudaMalloc(&d_kernel_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sigma_color, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sigma_space, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_threshold, sizeof(float)));

    // Copy kernel size, sigma_color, sigma_space and threshold to device memory
    CUDA_CHECK(cudaMemcpy(d_kernel_size, &kernel_size, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigma_color, &sigma_color, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigma_space, &sigma_space, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_threshold, &threshold, sizeof(float), cudaMemcpyHostToDevice));

    // Launch the CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    bilateral_filter_threshold_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, 
        d_output, 
        d_kernel_size, 
        d_sigma_color, 
        d_sigma_space,
        d_threshold, 
        height, 
        width, 
        channels
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output tensor from device memory
    CUDA_CHECK(cudaMemcpy(output_tensor, d_output, height * width * channels * sizeof(half), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel_size));
    CUDA_CHECK(cudaFree(d_sigma_color));
    CUDA_CHECK(cudaFree(d_sigma_space));
    CUDA_CHECK(cudaFree(d_threshold));
}
}
