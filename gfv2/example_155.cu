
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for low-rank token mixing with bfloat16
__global__ void low_rank_token_mixing_bf16_kernel(
    const float* input_tensor, const float* weight, const float* bias, float* output,
    int batch_size, int seq_len, int hidden_dim, int num_heads, int head_dim) 
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch_size && seq_idx < seq_len) {
        for (int head_idx = 0; head_idx < num_heads; head_idx++) {
            float sum = 0.0f;
            for (int hidden_idx = 0; hidden_idx < hidden_dim; hidden_idx++) {
                __nv_bfloat16 a = float_to_bfloat16(input_tensor[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + hidden_idx]);
                __nv_bfloat16 b = float_to_bfloat16(weight[head_idx * head_dim * hidden_dim + hidden_idx]);
                sum += bfloat16_to_float(__hmul(a, b));
            }

            int output_idx = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + head_idx * head_dim;
            output[output_idx] = sum + bias[head_idx * head_dim];
            output[output_idx] = fmaxf(output[output_idx], 0.0f);
        }
    }
}

extern "C" {

void low_rank_token_mixing_bf16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);

    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);
    int bias_dim1 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    // Extract metadata
    int batch_size = input_tensor_dim0;
    int seq_len = input_tensor_dim1;
    int hidden_dim = input_tensor_dim2;
    int num_heads = weight_dim0;
    int head_dim = weight_dim1;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_weight, num_heads * head_dim * hidden_dim * sizeof(float));
    cudaMalloc(&d_bias, num_heads * head_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * seq_len * hidden_dim * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, num_heads * head_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, num_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y);

    low_rank_token_mixing_bf16_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output, 
        batch_size, seq_len, hidden_dim, num_heads, head_dim
    );

    // Copy results back to host
    cudaMemcpy(output, d_output, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
