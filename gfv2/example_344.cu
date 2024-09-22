
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for adaptive average pooling 1D
__global__ void adaptive_avg_pool1d_kernel(const half* input, half* output, int batch_size, int input_channels, int input_length) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && c < input_channels) {
        float sum = 0.0f;
        for (int i = 0; i < input_length; ++i) {
            sum += __int2float_rn(input[b * input_channels * input_length + c * input_length + i]);
        }
        output[b * input_channels + c] = __float2half_rn(sum / input_length);
    }
}

// CUDA kernel for einsum outer product and ReLU
__global__ void einsum_relu_kernel(const half* pooled, const half* weights, half* output, int batch_size, int input_channels, int output_channels) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < batch_size && c < input_channels && k < output_channels) {
        output[b * input_channels * output_channels + c * output_channels + k] = __int2half_rn(__float2half_rn(__float2half_rn(pooled[b * input_channels + c]) * __int2float_rn(weights[c * output_channels + k])) * __int2float_rn(__float2half_rn(pooled[b * input_channels + c]) * __int2float_rn(weights[c * output_channels + k])));
    }
}

extern "C" {

void my_fp16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract weights tensor
    const float* weights = va_arg(args, const float*);
    int weights_dim0 = va_arg(args, int);
    int weights_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    half* output = va_arg(args, half*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_channels = input_tensor_dim1;
    int input_length = input_tensor_dim2;
    int output_channels = weights_dim0;

    // Allocate device memory
    half *d_input, *d_weights, *d_pooled, *d_output;
    cudaMalloc(&d_input, batch_size * input_channels * input_length * sizeof(half));
    cudaMalloc(&d_weights, weights_dim0 * weights_dim1 * sizeof(half));
    cudaMalloc(&d_pooled, batch_size * input_channels * sizeof(half));
    cudaMalloc(&d_output, batch_size * input_channels * output_channels * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_channels * input_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, weights_dim0 * weights_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch adaptive average pooling kernel
    dim3 threadsPerBlock(32, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (input_channels + threadsPerBlock.y - 1) / threadsPerBlock.y);

    adaptive_avg_pool1d_kernel<<<numBlocks, threadsPerBlock>>>(
        reinterpret_cast<const half*>(d_input), d_pooled, batch_size, input_channels, input_length
    );

    // Launch einsum outer product and ReLU kernel
    threadsPerBlock = dim3(16, 16, 16);
    numBlocks = dim3((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (output_channels + threadsPerBlock.z - 1) / threadsPerBlock.z);

    einsum_relu_kernel<<<numBlocks, threadsPerBlock>>>(
        d_pooled, d_weights, d_output, batch_size, input_channels, output_channels
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_channels * output_channels * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_pooled);
    cudaFree(d_output);
}

}  // extern "C"
