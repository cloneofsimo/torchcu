
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper functions for bfloat16 conversion
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for Flash Attention (BF16)
__global__ void flash_attn_bf16_kernel(const float* query, const float* key, const float* value, const bool* mask, 
                                       float* output, int batch_size, int seq_len, int head_dim, int mask_len) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && i < seq_len) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            __nv_bfloat16 q = float_to_bfloat16(query[b * seq_len * head_dim + i * head_dim + j]);
            __nv_bfloat16 k = float_to_bfloat16(key[b * seq_len * head_dim + j * head_dim + i]); // Transposed access
            __nv_bfloat16 v = float_to_bfloat16(value[b * seq_len * head_dim + j * head_dim + i]);

            __nv_bfloat16 attn_score = __hmul(q, k) / sqrtf(float_to_bfloat16(head_dim));

            // Apply mask
            if (mask && !mask[b * mask_len + j]) {
                attn_score = float_to_bfloat16(-INFINITY);
            }

            __nv_bfloat16 softmax_score = expf(attn_score); // Softmax approximation

            sum += bfloat16_to_float(__hmul(softmax_score, v));
        }
        output[b * seq_len * head_dim + i * head_dim + j] = sum;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* query = va_arg(args, const float*);
    int query_dim0 = va_arg(args, int);
    int query_dim1 = va_arg(args, int);
    int query_dim2 = va_arg(args, int);

    const float* key = va_arg(args, const float*);
    int key_dim0 = va_arg(args, int);
    int key_dim1 = va_arg(args, int);
    int key_dim2 = va_arg(args, int);

    const float* value = va_arg(args, const float*);
    int value_dim0 = va_arg(args, int);
    int value_dim1 = va_arg(args, int);
    int value_dim2 = va_arg(args, int);

    // Extract mask (if available)
    const bool* mask = va_arg(args, const bool*);
    int mask_dim0 = va_arg(args, int);
    int mask_dim1 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = query_dim0;
    int seq_len = query_dim1;
    int head_dim = query_dim2;

    // Allocate device memory
    float *d_query, *d_key, *d_value, *d_output;
    bool *d_mask = nullptr;

    cudaMalloc(&d_query, batch_size * seq_len * head_dim * sizeof(float));
    cudaMalloc(&d_key, batch_size * seq_len * head_dim * sizeof(float));
    cudaMalloc(&d_value, batch_size * seq_len * head_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * seq_len * head_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_query, query, batch_size * seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key, batch_size * seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, batch_size * seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Copy mask to device if available
    if (mask) {
        cudaMalloc(&d_mask, batch_size * mask_dim1 * sizeof(bool));
        cudaMemcpy(d_mask, mask, batch_size * mask_dim1 * sizeof(bool), cudaMemcpyHostToDevice);
    }

    // Launch kernel
    dim3 threadsPerBlock(32, 32); // Adjust for optimal performance
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y);

    flash_attn_bf16_kernel<<<numBlocks, threadsPerBlock>>>(d_query, d_key, d_value, d_mask, d_output, 
                                                       batch_size, seq_len, head_dim, mask_dim1);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
    if (mask) {
        cudaFree(d_mask);
    }
}

}  // extern "C"
