
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

extern "C" {

__global__ void zero_padding_kernel(const float* input, float* output, int batch, int height, int width, int padding) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height + 2 * padding && col < width + 2 * padding) {
        int row_in = row - padding;
        int col_in = col - padding;

        if (row_in >= 0 && row_in < height && col_in >= 0 && col_in < width) {
            output[row * (width + 2 * padding) + col] = input[row_in * width + col_in];
        } else {
            output[row * (width + 2 * padding) + col] = 0.0f;
        }
    }
}

void zero_padding_fp32_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int batch = va_arg(args, int);
    int height = va_arg(args, int);
    int width = va_arg(args, int);

    // Extract padding value
    int padding = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch * height * width * sizeof(float));
    cudaMalloc(&d_output, batch * (height + 2 * padding) * (width + 2 * padding) * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 2 * padding + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + 2 * padding + threadsPerBlock.y - 1) / threadsPerBlock.y);

    zero_padding_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, batch, height, width, padding);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch * (height + 2 * padding) * (width + 2 * padding) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

} // extern "C"
