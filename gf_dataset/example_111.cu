
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

__global__ void flash_attention_kernel_fp16(
    const half* query, const half* key, const half* value, const bool* mask, half* output,
    int batch, int head, int seq_len, int head_dim
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int q = threadIdx.z;

    if (b < batch && h < head && q < seq_len) {
        float sum = 0.0f;
        for (int k = 0; k < seq_len; ++k) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += __float2half_rn(__hmul(__half2float(query[b * head * seq_len * head_dim + h * seq_len * head_dim + q * head_dim + d]),
                                             __half2float(key[b * head * seq_len * head_dim + h * seq_len * head_dim + k * head_dim + d])));
            }
            score /= sqrtf(head_dim);

            if (mask != nullptr && !mask[b * seq_len + k]) {
                score = -INFINITY;
            }

            sum += __half2float(expf(score));
        }

        for (int k = 0; k < seq_len; ++k) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += __float2half_rn(__hmul(__half2float(query[b * head * seq_len * head_dim + h * seq_len * head_dim + q * head_dim + d]),
                                             __half2float(key[b * head * seq_len * head_dim + h * seq_len * head_dim + k * head_dim + d])));
            }
            score /= sqrtf(head_dim);

            if (mask != nullptr && !mask[b * seq_len + k]) {
                score = -INFINITY;
            }

            float prob = expf(score) / sum;
            
            for (int d = 0; d < head_dim; ++d) {
                output[b * head * seq_len * head_dim + h * seq_len * head_dim + q * head_dim + d] = __float2half_rn(prob * __half2float(value[b * head * seq_len * head_dim + h * seq_len * head_dim + k * head_dim + d]));
            }
        }
    }
}

extern "C" {
void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* query_data = va_arg(args, const float*);
    int query_dim0 = va_arg(args, int);
    int query_dim1 = va_arg(args, int);
    int query_dim2 = va_arg(args, int);
    int query_dim3 = va_arg(args, int);

    const float* key_data = va_arg(args, const float*);
    int key_dim0 = va_arg(args, int);
    int key_dim1 = va_arg(args, int);
    int key_dim2 = va_arg(args, int);
    int key_dim3 = va_arg(args, int);

    const float* value_data = va_arg(args, const float*);
    int value_dim0 = va_arg(args, int);
    int value_dim1 = va_arg(args, int);
    int value_dim2 = va_arg(args, int);
    int value_dim3 = va_arg(args, int);

    const bool* mask_data = va_arg(args, const bool*);
    int mask_dim0 = va_arg(args, int);
    int mask_dim1 = va_arg(args, int);

    float* output_data = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    half *d_query, *d_key, *d_value, *d_output;
    bool *d_mask = nullptr;
    cudaMalloc(&d_query, query_dim0 * query_dim1 * query_dim2 * query_dim3 * sizeof(half));
    cudaMalloc(&d_key, key_dim0 * key_dim1 * key_dim2 * key_dim3 * sizeof(half));
    cudaMalloc(&d_value, value_dim0 * value_dim1 * value_dim2 * value_dim3 * sizeof(half));
    cudaMalloc(&d_output, query_dim0 * query_dim1 * query_dim2 * query_dim3 * sizeof(half));
    if (mask_data) {
        cudaMalloc(&d_mask, mask_dim0 * mask_dim1 * sizeof(bool));
    }

    // Copy input data to device
    cudaMemcpy(d_query, query_data, query_dim0 * query_dim1 * query_dim2 * query_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key_data, key_dim0 * key_dim1 * key_dim2 * key_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value_data, value_dim0 * value_dim1 * value_dim2 * value_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    if (mask_data) {
        cudaMemcpy(d_mask, mask_data, mask_dim0 * mask_dim1 * sizeof(bool), cudaMemcpyHostToDevice);
    }

    // Launch kernel
    dim3 threadsPerBlock(32, 1, 1);
    dim3 numBlocks(
        (query_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (query_dim1 + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (query_dim2 + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    flash_attention_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_query, d_key, d_value, d_mask, d_output, query_dim0, query_dim1, query_dim2, query_dim3
    );

    // Copy result back to host
    cudaMemcpy(output_data, d_output, query_dim0 * query_dim1 * query_dim2 * query_dim3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
    if (d_mask) {
        cudaFree(d_mask);
    }
}
}
