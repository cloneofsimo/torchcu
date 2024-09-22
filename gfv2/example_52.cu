
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

// CUDA kernel for max pooling with fading out
__global__ void max_pool_fading_kernel_bf16(const float* input, float* output, int batch_size, int channels, int input_height, int input_width,
                                             int kernel_size, int stride, float fading_factor) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;

    if (row < input_height / stride && col < input_width / stride && b < batch_size) {
        int output_row = row * stride;
        int output_col = col * stride;
        float max_value = -FLT_MAX;

        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int input_row = output_row + i;
                int input_col = output_col + j;

                if (input_row < input_height && input_col < input_width) {
                    __nv_bfloat16 val = float_to_bfloat16(input[((b * channels * input_height + input_row) * input_width + input_col)]);
                    max_value = fmaxf(max_value, bfloat16_to_float(val));
                }
            }
        }
        output[((b * channels * input_height / stride + row) * input_width / stride + col)] = max_value * (1 - fading_factor);
    }
}

extern "C" {

void max_pool_bf16_fading(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int channels = va_arg(args, int);
    int input_height = va_arg(args, int);
    int input_width = va_arg(args, int);

    // Extract kernel_size
    int kernel_size = va_arg(args, int);

    // Extract stride
    int stride = va_arg(args, int);

    // Extract fading factor
    float fading_factor = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * input_height / stride * input_width / stride * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_width / stride + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (input_height / stride + threadsPerBlock.y - 1) / threadsPerBlock.y, batch_size);

    max_pool_fading_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, channels, input_height, input_width,
        kernel_size, stride, fading_factor
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * input_height / stride * input_width / stride * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
