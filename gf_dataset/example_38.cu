
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// CUDA kernel for grid sampling and SELU activation
__global__ void grid_sample_selu_kernel(const float* input_tensor, const float* grid, int8_t* output,
                                        int batch_size, int channels, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            // Get grid coordinates
            float grid_x = grid[((y * width + x) * 2 + 0) + c];
            float grid_y = grid[((y * width + x) * 2 + 1) + c];

            // Bilinear interpolation
            int x0 = floorf(grid_x);
            int y0 = floorf(grid_y);
            int x1 = ceilf(grid_x);
            int y1 = ceilf(grid_y);

            float weight_x0 = x1 - grid_x;
            float weight_x1 = grid_x - x0;
            float weight_y0 = y1 - grid_y;
            float weight_y1 = grid_y - y0;

            // Check boundary conditions
            x0 = max(0, min(x0, width - 1));
            x1 = max(0, min(x1, width - 1));
            y0 = max(0, min(y0, height - 1));
            y1 = max(0, min(y1, height - 1));

            // Accumulate weighted values
            sum += weight_x0 * weight_y0 * input_tensor[((y0 * width + x0) * channels + c) + (batch_size * height * width * channels)];
            sum += weight_x1 * weight_y0 * input_tensor[((y0 * width + x1) * channels + c) + (batch_size * height * width * channels)];
            sum += weight_x0 * weight_y1 * input_tensor[((y1 * width + x0) * channels + c) + (batch_size * height * width * channels)];
            sum += weight_x1 * weight_y1 * input_tensor[((y1 * width + x1) * channels + c) + (batch_size * height * width * channels)];
        }

        // Apply SELU activation
        sum = 1.0507009873554804934193349852946 * (sum + 1.6732632423543772848170429916717 * (sum > 0) - 1.6732632423543772848170429916717 * (sum <= 0));

        // Quantize to int8
        output[((y * width + x) * channels) + (batch_size * height * width * channels)] = __float2int_rn(sum);
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

    // Extract grid tensor
    const float* grid = va_arg(args, const float*);
    int grid_dim0 = va_arg(args, int);
    int grid_dim1 = va_arg(args, int);
    int grid_dim2 = va_arg(args, int);
    int grid_dim3 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    int8_t* output = va_arg(args, int8_t*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;

    // Allocate device memory
    float *d_input, *d_grid;
    int8_t *d_output;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_grid, grid_dim0 * grid_dim1 * grid_dim2 * grid_dim3 * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * height * width * sizeof(int8_t));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid, grid, grid_dim0 * grid_dim1 * grid_dim2 * grid_dim3 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    grid_sample_selu_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_grid, d_output, batch_size, channels, height, width
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * height * width * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_grid);
    cudaFree(d_output);
}

}  // extern "C"
