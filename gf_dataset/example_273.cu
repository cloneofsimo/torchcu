
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// CUDA kernel for coordinate convolution with adaptive max pooling
__global__ void coord_conv_kernel(const float* input_tensor, const int* filter_size, float* output,
                                  int B, int C, int H, int W, int filter_size_val) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < H && col < W) {
        // Calculate coordinates
        int x = col;
        int y = row;

        // Initialize max value and index
        float max_val = -FLT_MAX;
        int max_index = -1;

        // Iterate over kernel window
        for (int i = -filter_size_val / 2; i <= filter_size_val / 2; ++i) {
            for (int j = -filter_size_val / 2; j <= filter_size_val / 2; ++j) {
                int kernel_row = y + i;
                int kernel_col = x + j;

                // Check boundaries
                if (kernel_row >= 0 && kernel_row < H && kernel_col >= 0 && kernel_col < W) {
                    int input_index = (row * W + col) * C + (kernel_row * W + kernel_col) * (C + 2);
                    float val = input_tensor[input_index];
                    if (val > max_val) {
                        max_val = val;
                        max_index = (kernel_row * W + kernel_col) * C;
                    }
                }
            }
        }

        // Write output value
        output[(row * W + col) * C] = max_val;
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

    // Extract filter_size
    const int* filter_size = va_arg(args, const int*);
    int filter_size_val = filter_size[0];

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int B = input_tensor_dim0;
    int C = input_tensor_dim1;
    int H = input_tensor_dim2;
    int W = input_tensor_dim3;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, B * C * H * W * sizeof(float));
    cudaMalloc(&d_output, B * C * (H / 2) * (W / 2) * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((W / 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (H / 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    coord_conv_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, filter_size, d_output, B, C, H, W, filter_size_val
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, B * C * (H / 2) * (W / 2) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
