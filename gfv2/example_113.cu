
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for log filter
__global__ void log_filter_kernel(const float* input_tensor, float* output, int batch, int channels, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = (y * width + x) * channels;

    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            output[index + c] = logf(input_tensor[index + c] + 1e-6f);
        }
    }
}

// CUDA kernel for generating affine grid
__global__ void affine_grid_generator_kernel(const float* theta, float* grid, int batch, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = (y * width + x) * 3;

    if (x < width && y < height) {
        float x_norm = (2.0f * x / (width - 1) - 1.0f);
        float y_norm = (2.0f * y / (height - 1) - 1.0f);
        float z_norm = 1.0f;

        // Affine transformation using theta
        grid[index] = x_norm * theta[batch * 6] + y_norm * theta[batch * 6 + 1] + theta[batch * 6 + 2];
        grid[index + 1] = x_norm * theta[batch * 6 + 3] + y_norm * theta[batch * 6 + 4] + theta[batch * 6 + 5];
        grid[index + 2] = z_norm;
    }
}

extern "C" {

void log_filter_affine_grid_generator(int num_args, ...) {
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

    // Extract output tensor (assuming it's preallocated)
    float* grid = va_arg(args, float*);

    va_end(args);

    int batch = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;

    // Allocate device memory
    float *d_input, *d_theta, *d_grid;
    cudaMalloc(&d_input, batch * channels * height * width * sizeof(float));
    cudaMalloc(&d_theta, theta_dim0 * theta_dim1 * sizeof(float));
    cudaMalloc(&d_grid, batch * channels * height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, theta, theta_dim0 * theta_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch log filter kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    log_filter_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_grid, batch, channels, height, width);

    // Launch affine grid generator kernel
    affine_grid_generator_kernel<<<numBlocks, threadsPerBlock>>>(d_theta, d_grid, batch, height, width);

    // Copy result back to host
    cudaMemcpy(grid, d_grid, batch * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_theta);
    cudaFree(d_grid);
}

}  // extern "C"
