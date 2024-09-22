
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

// CUDA kernel for pixel unshuffle with layer scaling
__global__ void pixel_unshuffle_kernel_fp16(const float* input_tensor, float scale, float* output, 
                                        int batch_size, int channels, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int output_width = width * 2;
    int output_height = height * 2;

    if (x < output_width && y < output_height) {
        int input_x = x / 2;
        int input_y = y / 2;
        int input_index = (batch_size * channels * input_y + input_x) * height * width;
        int output_index = (batch_size * channels * y + x) * output_height * output_width;

        half input_value = float_to_half(input_tensor[input_index]);
        output[output_index] = half_to_float(input_value) * scale;
    }
}

extern "C" {

void pixel_unshuffle_fp16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract scale
    float scale = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * height * 2 * width * 2 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width * 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height * 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    pixel_unshuffle_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_input, scale, d_output, batch_size, channels, height, width
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * height * 2 * width * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
