
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for multi-head attention
__global__ void multihead_attention_kernel(const float* query, const float* key, const float* value, 
                                         float* output, int batch_size, int seq_len, int hidden_dim, int num_heads) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int head_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && head_idx < num_heads && seq_idx < seq_len) {
        int hidden_dim_per_head = hidden_dim / num_heads;

        // Calculate attention scores for the current head
        float attention_scores[seq_len];
        for (int i = 0; i < seq_len; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < hidden_dim_per_head; ++j) {
                sum += query[batch_idx * num_heads * seq_len * hidden_dim_per_head + 
                             head_idx * seq_len * hidden_dim_per_head + seq_idx * hidden_dim_per_head + j] * 
                       key[batch_idx * num_heads * seq_len * hidden_dim_per_head + 
                             head_idx * seq_len * hidden_dim_per_head + i * hidden_dim_per_head + j];
            }
            attention_scores[i] = sum / sqrtf(hidden_dim_per_head);
        }

        // Apply softmax to get attention weights
        float attention_weights[seq_len];
        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            attention_weights[i] = expf(attention_scores[i]);
            sum_exp += attention_weights[i];
        }
        for (int i = 0; i < seq_len; ++i) {
            attention_weights[i] /= sum_exp;
        }

        // Perform weighted sum of value
        for (int j = 0; j < hidden_dim_per_head; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < seq_len; ++i) {
                sum += attention_weights[i] * value[batch_idx * num_heads * seq_len * hidden_dim_per_head + 
                                                   head_idx * seq_len * hidden_dim_per_head + i * hidden_dim_per_head + j];
            }
            output[batch_idx * num_heads * seq_len * hidden_dim_per_head + 
                   head_idx * seq_len * hidden_dim_per_head + seq_idx * hidden_dim_per_head + j] = sum;
        }
    }
}

extern "C" {

void multihead_attention_fp32(int num_args, ...) {
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

    int num_heads = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = query_dim0;
    int seq_len = query_dim1;
    int hidden_dim = query_dim2;

    // Allocate device memory
    float *d_query, *d_key, *d_value, *d_output;
    cudaMalloc(&d_query, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_key, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_value, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * seq_len * hidden_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_query, query, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 8, 1);
    dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (num_heads + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    multihead_attention_kernel<<<numBlocks, threadsPerBlock>>>(
        d_query, d_key, d_value, d_output, batch_size, seq_len, hidden_dim, num_heads
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
}

}  // extern "C"
