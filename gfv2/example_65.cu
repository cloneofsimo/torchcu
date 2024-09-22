
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for self-attention inplace
__global__ void self_attention_inplace_kernel(float* query, float* key, float* value,
                                          int batch_size, int seq_len, int d_model) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && i < seq_len) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            float score = 0.0f;
            for (int k = 0; k < d_model; ++k) {
                score += query[b * seq_len * d_model + i * d_model + k] *
                         key[b * seq_len * d_model + j * d_model + k];
            }
            score /= sqrtf((float)d_model);
            sum += expf(score);
        }

        // Calculate attention weights
        for (int j = 0; j < seq_len; ++j) {
            float score = 0.0f;
            for (int k = 0; k < d_model; ++k) {
                score += query[b * seq_len * d_model + i * d_model + k] *
                         key[b * seq_len * d_model + j * d_model + k];
            }
            score /= sqrtf((float)d_model);
            float attention = expf(score) / sum;

            // Update query with weighted value
            for (int k = 0; k < d_model; ++k) {
                query[b * seq_len * d_model + i * d_model + k] += 
                    attention * value[b * seq_len * d_model + j * d_model + k];
            }
        }
    }
}

extern "C" {
void self_attention_inplace(int num_args, ...) {
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

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((query_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (query_dim1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    self_attention_inplace_kernel<<<numBlocks, threadsPerBlock>>>(
        const_cast<float*>(query), const_cast<float*>(key), const_cast<float*>(value),
        query_dim0, query_dim1, query_dim2
    );

    cudaDeviceSynchronize();
}
}
