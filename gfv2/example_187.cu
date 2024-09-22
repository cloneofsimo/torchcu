
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for matrix multiplication, addition, pixel unshuffle, ReLU, and comparison
__global__ void custom_kernel(const half* input_tensor, const half* weight, const half* bias, half* output, 
                               int m, int n, int k, int downscale_factor) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = row / downscale_factor;
    int out_col = col / downscale_factor;

    if (row < m && col < n) {
        // Matrix multiplication
        half sum = make_half(0.0f);
        for (int i = 0; i < k; ++i) {
            sum += __hmul(input_tensor[row * k + i], weight[col * k + i]);
        }

        // Addition
        sum += bias[col];

        // Pixel unshuffle
        output[out_row * (n / downscale_factor) + out_col] = sum;

        // ReLU activation
        output[out_row * (n / downscale_factor) + out_col] = __fmax(output[out_row * (n / downscale_factor) + out_col], make_half(0.0f));

        // Less than comparison
        output[out_row * (n / downscale_factor) + out_col] = __hmul(output[out_row * (n / downscale_factor) + out_col], __hgt(output[out_row * (n / downscale_factor) + out_col], make_half(0.5f)));
    }
}

extern "C" {

void custom_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;
    int downscale_factor = 2;

    // Allocate device memory
    half *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(half));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(half));
    cudaMalloc(&d_bias, output_dim * sizeof(half));
    cudaMalloc(&d_output, (batch_size / downscale_factor) * (output_dim / downscale_factor) * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_dim * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim / downscale_factor + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size / downscale_factor + threadsPerBlock.y - 1) / threadsPerBlock.y);

    custom_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output, batch_size, output_dim, input_dim, downscale_factor
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, (batch_size / downscale_factor) * (output_dim / downscale_factor) * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
