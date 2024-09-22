
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

// CUDA kernel for 1D convolution using bfloat16
__global__ void conv1d_kernel_bf16(const float* input, const float* weight, const float* bias, float* output, 
                                    int batch_size, int in_channels, int out_channels, int kernel_size, int input_length) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = batch_idx * out_channels + out_channel_idx;

    if (batch_idx < batch_size && out_channel_idx < out_channels) {
        float sum = bias[out_channel_idx];  // Initialize with bias
        for (int in_channel_idx = 0; in_channel_idx < in_channels; ++in_channel_idx) {
            for (int k = 0; k < kernel_size; ++k) {
                int input_idx = batch_idx * in_channels * input_length + in_channel_idx * input_length + k;
                int weight_idx = out_channel_idx * in_channels * kernel_size + in_channel_idx * kernel_size + k;
                if (input_idx < batch_size * in_channels * input_length && weight_idx < out_channels * in_channels * kernel_size) {
                    __nv_bfloat16 a = float_to_bfloat16(input[input_idx]);
                    __nv_bfloat16 b = float_to_bfloat16(weight[weight_idx]);
                    sum += bfloat16_to_float(__hmul(a, b));
                }
            }
        }
        output[out_idx] = bfloat16_to_float(__logsigmoid_rn(float_to_bfloat16(sum)));  // LogSigmoid activation
    }
}

// CUDA kernel for log sigmoid activation using bfloat16
__global__ void logsigmoid_kernel_bf16(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = bfloat16_to_float(__logsigmoid_rn(float_to_bfloat16(input[idx])));
    }
}

extern "C" {

void conv_logsigmoid_bf16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int input_length = input_tensor_dim2;
    int out_channels = weight_dim0;
    int kernel_size = weight_dim2;
    int output_length = input_length - kernel_size + 1;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * in_channels * input_length * sizeof(float));
    cudaMalloc(&d_weight, out_channels * in_channels * kernel_size * sizeof(float));
    cudaMalloc(&d_bias, out_channels * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * output_length * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * in_channels * input_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, out_channels * in_channels * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Launch convolution kernel
    dim3 threadsPerBlock(32, 1);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (out_channels + threadsPerBlock.y - 1) / threadsPerBlock.y);
    conv1d_kernel_bf16<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_bias, d_output, 
                                                    batch_size, in_channels, out_channels, kernel_size, input_length);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * out_channels * output_length * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
