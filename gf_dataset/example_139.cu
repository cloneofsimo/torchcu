
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for generating affine grid
__global__ void affine_grid_kernel(const float* theta, float* grid, 
                                    int batch_size, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height) {
        int flat_idx = (idy * width + idx) * batch_size;

        float x = (float)idx / (width - 1) * 2.0f - 1.0f;
        float y = (float)idy / (height - 1) * 2.0f - 1.0f;

        // Apply affine transformation
        grid[flat_idx] = theta[0] * x + theta[1] * y + theta[2];
        grid[flat_idx + 1] = theta[3] * x + theta[4] * y + theta[5];
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

    // Extract theta tensor
    const float* theta = va_arg(args, const float*);
    int theta_dim0 = va_arg(args, int);
    int theta_dim1 = va_arg(args, int);
    int theta_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;

    // Allocate device memory
    float *d_theta, *d_output;
    cudaMalloc(&d_theta, theta_dim0 * theta_dim1 * theta_dim2 * sizeof(float));
    cudaMalloc(&d_output, batch_size * height * width * 2 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_theta, theta, theta_dim0 * theta_dim1 * theta_dim2 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    affine_grid_kernel<<<numBlocks, threadsPerBlock>>>(
        d_theta, d_output, batch_size, height, width
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * height * width * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_theta);
    cudaFree(d_output);
}

}  // extern "C"
