
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for median filtering
__global__ void median_filter_kernel(const float* input, float* output, int height, int width, int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int half_kernel = kernel_size / 2;
        int start_row = max(0, row - half_kernel);
        int end_row = min(height, row + half_kernel + 1);
        int start_col = max(0, col - half_kernel);
        int end_col = min(width, col + half_kernel + 1);

        int kernel_size_sq = (end_row - start_row) * (end_col - start_col);
        float kernel_data[kernel_size_sq];

        int k = 0;
        for (int r = start_row; r < end_row; ++r) {
            for (int c = start_col; c < end_col; ++c) {
                kernel_data[k++] = input[r * width + c];
            }
        }

        // Sort the kernel data using bubble sort (for small kernel sizes)
        for (int i = 0; i < kernel_size_sq - 1; ++i) {
            for (int j = 0; j < kernel_size_sq - i - 1; ++j) {
                if (kernel_data[j] > kernel_data[j + 1]) {
                    float temp = kernel_data[j];
                    kernel_data[j] = kernel_data[j + 1];
                    kernel_data[j + 1] = temp;
                }
            }
        }

        // Median is the middle element
        output[row * width + col] = kernel_data[kernel_size_sq / 2];
    }
}

extern "C" {

void median_filter(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);

    // Extract kernel size
    int kernel_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (input_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    median_filter_kernel<<<numBlocks, threadsPerBlock>>>(input, output, input_dim0, input_dim1, kernel_size);

    cudaDeviceSynchronize();
}

}  // extern "C"

