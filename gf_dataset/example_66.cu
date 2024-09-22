
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for affine grid generation
__global__ void affine_grid_generator_kernel_fp16(const float* input_tensor, const float* theta, half* output,
                                        int batch_size, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * width + col;

    if (row < height && col < width) {
        float x = (col + 0.5f) / width;
        float y = (row + 0.5f) / height;

        // Affine transformation
        float transformed_x = theta[0] * x + theta[1] * y + theta[2];
        float transformed_y = theta[3] * x + theta[4] * y + theta[5];

        // Map to [-1, 1] range
        transformed_x = (transformed_x * 2.0f) - 1.0f;
        transformed_y = (transformed_y * 2.0f) - 1.0f;

        // Store in output grid
        output[index * 2] = float_to_half(transformed_x);
        output[index * 2 + 1] = float_to_half(transformed_y);
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
    half* output = va_arg(args, half*);

    va_end(args);

    // Get dimensions
    int batch_size = input_tensor_dim0;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;

    // Allocate device memory
    float *d_input, *d_theta;
    half *d_output;
    cudaMalloc(&d_input, batch_size * input_tensor_dim1 * height * width * sizeof(float));
    cudaMalloc(&d_theta, theta_dim0 * theta_dim1 * theta_dim2 * sizeof(float));
    cudaMalloc(&d_output, batch_size * height * width * 2 * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_tensor_dim1 * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, theta, theta_dim0 * theta_dim1 * theta_dim2 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    affine_grid_generator_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_theta, d_output, batch_size, height, width
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * height * width * 2 * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_theta);
    cudaFree(d_output);
}

}  // extern "C"
