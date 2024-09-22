
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function for float to half conversion
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function for half to float conversion
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for masked attention computation
__global__ void attention_mask_kernel(const float* query, const float* key, const bool* mask, float* attention_weights,
                                     int batch_size, int seq_len, int hidden_dim) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && i < seq_len) {
        float score = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            if (mask[b * seq_len + j]) {
                // Clamp query and key for fp16 safety
                float q = query[b * hidden_dim * seq_len + i * hidden_dim + j];
                float k = key[b * hidden_dim * seq_len + i * hidden_dim + j];
                q = fmaxf(fminf(q, 8.0f), -8.0f);
                k = fmaxf(fminf(k, 8.0f), -8.0f);

                // Compute score
                score += half_to_float(float_to_half(q) * float_to_half(k));
            }
        }

        // Softmax normalization
        float max_score = -INFINITY;
        for (int j = 0; j < seq_len; ++j) {
            if (mask[b * seq_len + j]) {
                float temp_score = 0.0f;
                for (int k = 0; k < seq_len; ++k) {
                    if (mask[b * seq_len + k]) {
                        temp_score += half_to_float(float_to_half(query[b * hidden_dim * seq_len + i * hidden_dim + k]) *
                                                       float_to_half(key[b * hidden_dim * seq_len + i * hidden_dim + k]));
                    }
                }
                max_score = fmaxf(max_score, temp_score);
            }
        }

        float exp_sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            if (mask[b * seq_len + j]) {
                float temp_score = 0.0f;
                for (int k = 0; k < seq_len; ++k) {
                    if (mask[b * seq_len + k]) {
                        temp_score += half_to_float(float_to_half(query[b * hidden_dim * seq_len + i * hidden_dim + k]) *
                                                       float_to_half(key[b * hidden_dim * seq_len + i * hidden_dim + k]));
                    }
                }
                exp_sum += expf(temp_score - max_score);
            }
        }

        attention_weights[b * seq_len * seq_len + i * seq_len + j] = expf(score - max_score) / exp_sum;
    }
}

extern "C" {

void attention_mask_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* query = va_arg(args, const float*);
    int query_dim0 = va_arg(args, int);
    int query_dim1 = va_arg(args, int);
    int query_dim2 = va_arg(args, int);

    const float* key = va_arg(args, const float*);
    int key_dim0 = va_arg(args, int);
    int key_dim1 = va_arg(args, int);
    int key_dim2 = va_arg(args, int);

    const bool* mask = va_arg(args, const bool*);
    int mask_dim0 = va_arg(args, int);
    int mask_dim1 = va_arg(args, int);

    float* attention_weights = va_arg(args, float*);

    va_end(args);

    int batch_size = query_dim0;
    int seq_len = query_dim1;
    int hidden_dim = query_dim2;

    // Allocate device memory
    float *d_query, *d_key, *d_attention_weights;
    bool *d_mask;
    cudaMalloc(&d_query, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_key, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_attention_weights, batch_size * seq_len * seq_len * sizeof(float));
    cudaMalloc(&d_mask, batch_size * seq_len * sizeof(bool));

    // Copy data to device
    cudaMemcpy(d_query, query, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, batch_size * seq_len * sizeof(bool), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y);

    attention_mask_kernel<<<numBlocks, threadsPerBlock>>>(
        d_query, d_key, d_mask, d_attention_weights, batch_size, seq_len, hidden_dim
    );

    // Copy result back to host
    cudaMemcpy(attention_weights, d_attention_weights, batch_size * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_mask);
    cudaFree(d_attention_weights);
}

}  // extern "C"
