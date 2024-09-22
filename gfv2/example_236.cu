
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for window attention with fp16
__global__ void window_attention_fp16_kernel(const float* query, const float* key, const float* value, 
                                              const bool* mask, float* output,
                                              int B, int N, int C, int head_dim,
                                              int num_heads, bool qkv_bias, float attn_drop, float proj_drop) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < B && n < N) {
        float sum = 0.0f;
        for (int h = 0; h < num_heads; ++h) {
            half q_val = float_to_half(query[(b * N + n) * C + h * head_dim]);
            half sum_attn = 0.0h;
            for (int k = 0; k < N; ++k) {
                half k_val = float_to_half(key[(b * N + k) * C + h * head_dim]);
                half attn_val = __hmul(q_val, k_val) * (1.0f / sqrtf(head_dim));
                if (mask != nullptr && !mask[b * N * N + n * N + k]) {
                    attn_val = -INFINITY;
                }
                sum_attn += __hexp(attn_val);
            }

            sum_attn = __hdiv(sum_attn, sum_attn + 1.0h);
            
            if (attn_drop > 0.0f) {
                float random_val = (float)rand() / RAND_MAX;
                if (random_val < attn_drop) {
                    sum_attn = 0.0h;
                }
            }

            for (int k = 0; k < N; ++k) {
                half k_val = float_to_half(key[(b * N + k) * C + h * head_dim]);
                half attn_val = __hmul(q_val, k_val) * (1.0f / sqrtf(head_dim));
                if (mask != nullptr && !mask[b * N * N + n * N + k]) {
                    attn_val = -INFINITY;
                }
                sum += half_to_float(sum_attn * __hmul(attn_val, float_to_half(value[(b * N + k) * C + h * head_dim])));
            }
        }
        output[b * N * C + n * C] = sum;
    }
}

extern "C" {

void window_attention_fp16(int num_args, ...) {
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

    const bool* mask = va_arg(args, const bool*);
    int mask_dim0 = va_arg(args, int);
    int mask_dim1 = va_arg(args, int);
    int mask_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    // Extract additional parameters
    int head_dim = va_arg(args, int);
    bool qkv_bias = va_arg(args, int) != 0;
    float attn_drop = va_arg(args, double);
    float proj_drop = va_arg(args, double);

    va_end(args);

    int B = query_dim0;
    int N = query_dim1;
    int C = query_dim2;
    int num_heads = C / head_dim;

    // Allocate device memory
    float *d_query, *d_key, *d_value, *d_output;
    bool *d_mask = nullptr;
    cudaMalloc(&d_query, B * N * C * sizeof(float));
    cudaMalloc(&d_key, B * N * C * sizeof(float));
    cudaMalloc(&d_value, B * N * C * sizeof(float));
    cudaMalloc(&d_output, B * N * C * sizeof(float));

    if (mask_dim0 > 0) {
        cudaMalloc(&d_mask, B * N * N * sizeof(bool));
        cudaMemcpy(d_mask, mask, B * N * N * sizeof(bool), cudaMemcpyHostToDevice);
    }

    // Copy input data to device
    cudaMemcpy(d_query, query, B * N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key, B * N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, B * N * C * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((B + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    window_attention_fp16_kernel<<<numBlocks, threadsPerBlock>>>(
        d_query, d_key, d_value, d_mask, d_output, B, N, C, head_dim, num_heads, qkv_bias, attn_drop, proj_drop
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, B * N * C * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
    if (mask_dim0 > 0) {
        cudaFree(d_mask);
    }
}

}  // extern "C"
