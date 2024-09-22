
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float half_to_float(half bf) {
    return __half2float(bf);
}

// CUDA kernel for box filter (average pooling)
__global__ void box_filter_kernel(const float* input_tensor, float* output_tensor, 
                                  int batch_size, int channels, int height, int width, 
                                  int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        float sum = 0.0f;
        for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i) {
            for (int j = -kernel_size / 2; j <= kernel_size / 2; ++j) {
                int r = row + i;
                int c = col + j;
                if (r >= 0 && r < height && c >= 0 && c < width) {
                    sum += input_tensor[(row * width + col) * channels + threadIdx.z];
                }
            }
        }
        output_tensor[(row * width + col) * channels + threadIdx.z] = sum / (kernel_size * kernel_size);
    }
}

extern "C" {

void torch_box_filter_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract kernel size
    int kernel_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output_tensor = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

    box_filter_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, channels, height, width, kernel_size
    );

    // Copy result back to host
    cudaMemcpy(output_tensor, d_output, batch_size * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
