
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for RReLU activation
__global__ void rrelu_kernel(const float* input, half* output, int batch_size, int channels, int height, int width,
                            float lower, float upper) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * channels * height * width) {
        int b = idx / (channels * height * width);
        int c = (idx % (channels * height * width)) / (height * width);
        int h = (idx % (height * width)) / width;
        int w = idx % width;

        float val = input[idx];
        float random_slope = lower + (upper - lower) * __float2half_rn(rand() / (float)RAND_MAX);  // Generate random slope
        output[idx] = float_to_half(val > 0 ? val : val * random_slope);
    }
}

// CUDA kernel for adaptive average pooling
__global__ void adaptive_avg_pool2d_kernel(const half* input, half* output, int batch_size, int channels, int height, int width,
                                          int output_height, int output_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * channels * output_height * output_width) {
        int b = idx / (channels * output_height * output_width);
        int c = (idx % (channels * output_height * output_width)) / (output_height * output_width);
        int oh = (idx % (output_height * output_width)) / output_width;
        int ow = idx % output_width;

        int input_height_start = oh * (height / output_height);
        int input_height_end = min((oh + 1) * (height / output_height), height);
        int input_width_start = ow * (width / output_width);
        int input_width_end = min((ow + 1) * (width / output_width), width);

        float sum = 0.0f;
        for (int h = input_height_start; h < input_height_end; ++h) {
            for (int w = input_width_start; w < input_width_end; ++w) {
                sum += half_to_float(input[b * channels * height * width + c * height * width + h * width + w]);
            }
        }
        output[idx] = float_to_half(sum / ((input_height_end - input_height_start) * (input_width_end - input_width_start)));
    }
}

// CUDA kernel for log operation
__global__ void log_kernel(const half* input, half* output, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * channels * height * width) {
        output[idx] = __logf(half_to_float(input[idx]));
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int channels = va_arg(args, int);
    int height = va_arg(args, int);
    int width = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    half *d_input, *d_output, *d_rrelu_output, *d_pool_output, *d_log_output;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(half));
    cudaMalloc(&d_output, batch_size * channels * height * width * sizeof(half));
    cudaMalloc(&d_rrelu_output, batch_size * channels * height * width * sizeof(half));
    cudaMalloc(&d_pool_output, batch_size * channels * 2 * 2 * sizeof(half));
    cudaMalloc(&d_log_output, batch_size * channels * 2 * 2 * sizeof(half));

    // Copy input data to device (converting to half)
    cudaMemcpy(d_input, input_tensor, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // RReLU
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size * channels * height * width + threadsPerBlock.x - 1) / threadsPerBlock.x);
    rrelu_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_rrelu_output, batch_size, channels, height, width, 0.125, 0.334);

    // Adaptive Average Pooling
    cudaMemcpy(d_pool_output, d_rrelu_output, batch_size * channels * height * width * sizeof(half), cudaMemcpyDeviceToDevice);
    adaptive_avg_pool2d_kernel<<<numBlocks, threadsPerBlock>>>(d_pool_output, d_output, batch_size, channels, height, width, 2, 2);

    // Log
    cudaMemcpy(d_log_output, d_output, batch_size * channels * 2 * 2 * sizeof(half), cudaMemcpyDeviceToDevice);
    log_kernel<<<numBlocks, threadsPerBlock>>>(d_log_output, d_output, batch_size, channels, 2, 2);

    // Copy result back to host (converting from half)
    cudaMemcpy(output, d_output, batch_size * channels * 2 * 2 * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_rrelu_output);
    cudaFree(d_pool_output);
    cudaFree(d_log_output);
}

}  // extern "C"
