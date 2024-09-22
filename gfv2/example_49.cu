
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for pruned, softshrink, and int8 quantized linear operation
__global__ void pruned_softshrink_int8_kernel(const float* input, const float* weight, float* output,
                                            int m, int n, int k, float prune_threshold) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int sum = 0;
        for (int i = 0; i < k; ++i) {
            float w = weight[col * k + i]; // Transposed access
            if (abs(w) > prune_threshold) {
                w = __int2float_rn(__float2int_rn(w * 0.5f)); // Soft shrink approximation
            } else {
                w = 0.0f;
            }
            sum += __int2float_rn(__float2int_rn(input[row * k + i]) * __float2int_rn(w));
        }
        output[row * n + col] = __int2float_rn(sum);
    }
}

extern "C" {
    void pruned_softshrink_int8_function(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        const float* input_tensor = va_arg(args, const float*);
        int input_tensor_dim0 = va_arg(args, int);
        int input_tensor_dim1 = va_arg(args, int);

        const float* weight = va_arg(args, const float*);
        int weight_dim0 = va_arg(args, int);
        int weight_dim1 = va_arg(args, int);

        float prune_threshold = va_arg(args, float);

        float* output = va_arg(args, float*);

        va_end(args);

        int batch_size = input_tensor_dim0;
        int input_dim = input_tensor_dim1;
        int output_dim = weight_dim0;

        // Allocate device memory
        float* d_input, *d_weight, *d_output;
        cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
        cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
        cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

        // Copy data to device
        cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        pruned_softshrink_int8_kernel<<<numBlocks, threadsPerBlock>>>(
            d_input, d_weight, d_output, batch_size, output_dim, input_dim, prune_threshold
        );

        // Copy result back to host
        cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_weight);
        cudaFree(d_output);
    }
}
