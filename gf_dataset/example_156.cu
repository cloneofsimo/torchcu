
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for box filter (max pooling)
__global__ void box_filter_max_kernel_bf16(const float* input, float* output, 
                                        int batch, int channels, int height, int width,
                                        int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    int c = threadIdx.z;

    if (row < height && col < width && b < batch && c < channels) {
        float max_val = -FLT_MAX;
        for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i) {
            for (int j = -kernel_size / 2; j <= kernel_size / 2; ++j) {
                int r = row + i;
                int c_ = col + j;
                if (r >= 0 && r < height && c_ >= 0 && c_ < width) {
                    __nv_bfloat16 val = float_to_bfloat16(input[b * channels * height * width + c * height * width + r * width + c_]);
                    max_val = fmaxf(max_val, bfloat16_to_float(val));
                }
            }
        }
        output[b * channels * height * width + c * height * width + row * width + col] = max_val;
    }
}

// CUDA kernel for box filter (min pooling)
__global__ void box_filter_min_kernel_bf16(const float* input, float* output, 
                                        int batch, int channels, int height, int width,
                                        int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    int c = threadIdx.z;

    if (row < height && col < width && b < batch && c < channels) {
        float min_val = FLT_MAX;
        for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i) {
            for (int j = -kernel_size / 2; j <= kernel_size / 2; ++j) {
                int r = row + i;
                int c_ = col + j;
                if (r >= 0 && r < height && c_ >= 0 && c_ < width) {
                    __nv_bfloat16 val = float_to_bfloat16(input[b * channels * height * width + c * height * width + r * width + c_]);
                    min_val = fminf(min_val, bfloat16_to_float(val));
                }
            }
        }
        output[b * channels * height * width + c * height * width + row * width + col] = min_val;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int batch = va_arg(args, int);
    int channels = va_arg(args, int);
    int height = va_arg(args, int);
    int width = va_arg(args, int);

    // Extract kernel size
    int kernel_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, batch * channels * height * width * sizeof(float));
    cudaMalloc(&d_output, batch * channels * height * width * sizeof(float));
    cudaMalloc(&d_temp, batch * channels * height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Dilation (max pooling)
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   batch);

    box_filter_max_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_temp, batch, channels, height, width, kernel_size
    );

    // Erosion (min pooling)
    box_filter_min_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_temp, d_output, batch, channels, height, width, kernel_size
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
}

}  // extern "C"
