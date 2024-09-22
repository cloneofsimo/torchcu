
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h> 

// CUDA kernel for box filtering
__global__ void box_filter_kernel(const float* input, float* output, int batch_size, int channels, 
                                    int height, int width, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    int b = blockIdx.w * blockDim.w + threadIdx.w;

    if (x < width && y < height && c < channels && b < batch_size) {
        float sum = 0.0f;
        for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i) {
            for (int j = -kernel_size / 2; j <= kernel_size / 2; ++j) {
                if (x + i >= 0 && x + i < width && y + j >= 0 && y + j < height) {
                    sum += input[((b * channels + c) * height + y + j) * width + x + i];
                }
            }
        }
        output[((b * channels + c) * height + y) * width + x] = sum / (kernel_size * kernel_size);
    }
}

// CUDA kernel for calculating Gaussian kernel weights
__global__ void gaussian_kernel_kernel(float* kernel, int kernel_size, float sigma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kernel_size) {
        float x = (i - kernel_size / 2);
        kernel[i] = expf(-(x * x) / (2 * sigma * sigma));
    }
}

// CUDA kernel for bilateral filtering
__global__ void bilateral_filter_kernel(const float* input, float* output, int batch_size, int channels,
                                         int height, int width, int kernel_size, float sigma_spatial, float sigma_color) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    int b = blockIdx.w * blockDim.w + threadIdx.w;

    if (x < width && y < height && c < channels && b < batch_size) {
        float sum = 0.0f;
        float weights_sum = 0.0f;
        float* spatial_kernel = new float[kernel_size];
        float* color_kernel = new float[kernel_size];

        // Calculate Gaussian kernels
        gaussian_kernel_kernel<<<1, kernel_size>>>(spatial_kernel, kernel_size, sigma_spatial);
        gaussian_kernel_kernel<<<1, kernel_size>>>(color_kernel, kernel_size, sigma_color);

        for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i) {
            for (int j = -kernel_size / 2; j <= kernel_size / 2; ++j) {
                if (x + i >= 0 && x + i < width && y + j >= 0 && y + j < height) {
                    // Calculate color distance
                    float color_dist = abs(input[((b * channels + c) * height + y + j) * width + x + i] - input[((b * channels + c) * height + y) * width + x]);
                    float color_weight = expf(-(color_dist * color_dist) / (2 * sigma_color * sigma_color));

                    // Calculate combined weight
                    float weight = spatial_kernel[i + kernel_size / 2] * spatial_kernel[j + kernel_size / 2] * color_weight;

                    // Accumulate weighted pixel value
                    sum += input[((b * channels + c) * height + y + j) * width + x + i] * weight;
                    weights_sum += weight;
                }
            }
        }

        // Normalize and set output
        output[((b * channels + c) * height + y) * width + x] = sum / weights_sum;

        delete[] spatial_kernel;
        delete[] color_kernel;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input image
    const float* input = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int channels = va_arg(args, int);
    int height = va_arg(args, int);
    int width = va_arg(args, int);

    // Extract kernel size
    int kernel_size = va_arg(args, int);

    // Extract sigma values
    float sigma_spatial = va_arg(args, float);
    float sigma_color = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory for input and output
    float* d_input, *d_output;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockDim(16, 16, 1); 
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 
                (channels + blockDim.z - 1) / blockDim.z, (batch_size + blockDim.w - 1) / blockDim.w);

    bilateral_filter_kernel<<<gridDim, blockDim>>>(d_input, d_output, batch_size, channels, height, width, kernel_size, sigma_spatial, sigma_color);

    // Copy the result back to host
    cudaMemcpy(output, d_output, batch_size * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

} // extern "C"
