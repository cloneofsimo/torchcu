
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for grid sampling with affine transformation
__global__ void grid_sampler_kernel(const float* input, const float* theta, float* output, 
                                      int batch_size, int channels, int height, int width) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && channel_idx < channels && row < height) {
        float x = (theta[batch_idx * 6 + 0] * row + theta[batch_idx * 6 + 1] * channel_idx + theta[batch_idx * 6 + 2]) / theta[batch_idx * 6 + 5];
        float y = (theta[batch_idx * 6 + 3] * row + theta[batch_idx * 6 + 4] * channel_idx + theta[batch_idx * 6 + 5]) / theta[batch_idx * 6 + 5];

        // Bilinear interpolation
        int x_floor = floor(x);
        int y_floor = floor(y);
        int x_ceil = ceil(x);
        int y_ceil = ceil(y);

        // Clamp indices to avoid out-of-bounds access
        x_floor = max(0, min(x_floor, width - 1));
        y_floor = max(0, min(y_floor, height - 1));
        x_ceil = max(0, min(x_ceil, width - 1));
        y_ceil = max(0, min(y_ceil, height - 1));

        float a = x - x_floor;
        float b = y - y_floor;

        // Access input tensor using indices
        int idx00 = (batch_idx * channels * height * width) + (channel_idx * height * width) + (y_floor * width) + x_floor;
        int idx01 = (batch_idx * channels * height * width) + (channel_idx * height * width) + (y_floor * width) + x_ceil;
        int idx10 = (batch_idx * channels * height * width) + (channel_idx * height * width) + (y_ceil * width) + x_floor;
        int idx11 = (batch_idx * channels * height * width) + (channel_idx * height * width) + (y_ceil * width) + x_ceil;

        float val00 = input[idx00];
        float val01 = input[idx01];
        float val10 = input[idx10];
        float val11 = input[idx11];

        // Bilinear interpolation calculation
        output[(batch_idx * channels * height * width) + (channel_idx * height * width) + (row * width) + channel_idx] = 
            (1 - a) * (1 - b) * val00 + a * (1 - b) * val01 + (1 - a) * b * val10 + a * b * val11;
    }
}

extern "C" {

void grid_sampler_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);
    int input_dim2 = va_arg(args, int);
    int input_dim3 = va_arg(args, int);

    // Extract theta tensor
    const float* theta = va_arg(args, const float*);
    int theta_dim0 = va_arg(args, int);
    int theta_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Dimensions
    int batch_size = input_dim0;
    int channels = input_dim1;
    int height = input_dim2;
    int width = input_dim3;

    // Allocate device memory
    float *d_input, *d_theta, *d_output;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_theta, theta_dim0 * theta_dim1 * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, theta, theta_dim0 * theta_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    grid_sampler_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_theta, d_output, batch_size, channels, height, width
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_theta);
    cudaFree(d_output);
}

}  // extern "C"
