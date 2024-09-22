
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

// CUDA kernel for SEBlock
__global__ void se_block_kernel(const float* input, float* output, int batch_size, int channels, 
                                int height, int width, int reduction) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && c < channels) {
        float sum = 0.0f;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                __nv_bfloat16 val = float_to_bfloat16(input[(b * channels + c) * height * width + h * width + w]);
                sum += bfloat16_to_float(val);
            }
        }
        sum /= height * width;

        __nv_bfloat16 reduced = float_to_bfloat16(sum);
        reduced = __hmul(reduced, reduced);  // Square
        reduced = __hmul(reduced, reduced);  // Square again
        reduced = __hmul(reduced, reduced);  // Square again
        reduced = __hmul(reduced, reduced);  // Square again

        // Apply sigmoid
        reduced = __hmul(reduced, __float2bfloat16(1.0f) + reduced);  // 1 + x
        reduced = __hmul(reduced, __float2bfloat16(0.5f));  // (1 + x) / 2

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                __nv_bfloat16 val = float_to_bfloat16(input[(b * channels + c) * height * width + h * width + w]);
                output[(b * channels + c) * height * width + h * width + w] = 
                    bfloat16_to_float(__hmul(val, reduced));
            }
        }
    }
}

// CUDA kernel for subtraction
__global__ void subtract_kernel(const float* se_output, const float* weight, float* output, 
                                int batch_size, int channels, int height, int width) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.z * blockDim.z + threadIdx.z;
    int w = threadIdx.w;

    if (b < batch_size && c < channels && h < height && w < width) {
        output[(b * channels + c) * height * width + h * width + w] =
            se_output[(b * channels + c) * height * width + h * width + w] - weight[c];
    }
}

extern "C" {

void se_subtract_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;
    int reduction = channels / 16; // Default reduction for SEBlock

    // Allocate device memory
    float *d_input, *d_se_output, *d_weight, *d_output;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_se_output, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_weight, channels * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, channels * sizeof(float), cudaMemcpyHostToDevice);

    // Launch SEBlock kernel
    dim3 se_block_threads(16, 16);
    dim3 se_block_blocks((channels + se_block_threads.y - 1) / se_block_threads.y, 
                           (batch_size + se_block_threads.x - 1) / se_block_threads.x);
    se_block_kernel<<<se_block_blocks, se_block_threads>>>(d_input, d_se_output, batch_size, channels, 
                                                         height, width, reduction);

    // Launch subtraction kernel
    dim3 subtract_threads(16, 16, 4);
    dim3 subtract_blocks((batch_size + subtract_threads.x - 1) / subtract_threads.x, 
                          (channels + subtract_threads.y - 1) / subtract_threads.y, 
                          (height + subtract_threads.z - 1) / subtract_threads.z);
    subtract_kernel<<<subtract_blocks, subtract_threads>>>(d_se_output, d_weight, d_output, 
                                                           batch_size, channels, height, width);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_se_output);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
