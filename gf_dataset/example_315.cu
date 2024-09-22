
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"

using namespace cutlass;

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for regularized grid sampling with layer scaling
__global__ void regularized_grid_sample_kernel(const float* input_tensor, const float* grid, float* output, 
                                            int batch_size, int channels, int height, int width, float scale) {
    int b = blockIdx.z;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && c < channels && h < height) {
        int grid_index = b * height * width * 2 + h * width * 2;
        float2 grid_coords = make_float2(grid[grid_index], grid[grid_index + 1]);

        // Clamp grid coordinates to prevent out-of-bounds access
        grid_coords.x = fmaxf(grid_coords.x, -1.0f);
        grid_coords.x = fminf(grid_coords.x, 1.0f);
        grid_coords.y = fmaxf(grid_coords.y, -1.0f);
        grid_coords.y = fminf(grid_coords.y, 1.0f);

        // Calculate normalized coordinates
        float2 normalized_coords = make_float2((grid_coords.x + 1.0f) / 2.0f, (grid_coords.y + 1.0f) / 2.0f);

        // Perform bilinear interpolation
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            int x_index = int(floor(normalized_coords.x * (width - 1)));
            int y_index = int(floor(normalized_coords.y * (height - 1)));
            x_index = clamp(x_index, 0, width - 1);
            y_index = clamp(y_index, 0, height - 1);

            float weight = (1.0f - normalized_coords.x) * (1.0f - normalized_coords.y);
            int input_index = b * channels * height * width + c * height * width + y_index * width + x_index;
            sum += weight * input_tensor[input_index];

            normalized_coords.x += 1.0f / (width - 1);
            normalized_coords.y += 1.0f / (height - 1);
        }

        output[b * channels * height * width + c * height * width + h * width] = sum * scale; 
    }
}

extern "C" {

void regularized_grid_sample(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    const float* grid = va_arg(args, const float*);
    int grid_dim0 = va_arg(args, int);
    int grid_dim1 = va_arg(args, int);
    int grid_dim2 = va_arg(args, int);
    int grid_dim3 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    // Extract scale value
    float scale = va_arg(args, float);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;

    // Allocate device memory
    float *d_input, *d_grid, *d_output;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_grid, batch_size * height * width * 2 * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid, grid, batch_size * height * width * 2 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   batch_size); 

    regularized_grid_sample_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_grid, d_output, batch_size, channels, height, width, scale
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_grid);
    cudaFree(d_output);
}

}  // extern "C"
