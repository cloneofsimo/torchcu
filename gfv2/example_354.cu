
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

#define CUDA_CHECK(condition)                                              \
{                                                                        \
    cudaError_t error = condition;                                         \
    if (error != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n",             \
                cudaGetErrorString(error), __FILE__, __LINE__);            \
        exit(EXIT_FAILURE);                                            \
    }                                                                    \
}

// Helper function to convert float to __half
__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert __half to float
__device__ __forceinline__ float half_to_float(__half hf) {
    return __half2float(hf);
}

// Helper function for the einsum operation
__device__ __forceinline__ float einsum_inner(float* x, float* y, int k, int i, int j) {
    float sum = 0.0f;
    for (int k = 0; k < k; ++k) {
        sum += x[i * k] * y[k * j];
    }
    return sum;
}

// Forward kernel for the multihead attention
__global__ void multihead_attention_forward_kernel(
    const float* query, const float* key, const float* value,
    const bool* attn_mask, const bool* key_padding_mask,
    const float* q_proj_weight, const float* k_proj_weight, const float* v_proj_weight,
    float* output, float* attn_weights,
    int batch_size, int seq_len, int num_heads, int head_dim,
    int query_stride, int key_stride, int value_stride, int output_stride, int attn_weights_stride
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int b = row / seq_len;
    int i = row % seq_len;
    int j = col;

    if (row < batch_size * seq_len && col < seq_len) {
        // Check key padding mask
        if (key_padding_mask[b * seq_len + j] == 1) {
            output[row * output_stride + col] = 0.0f;
            return;
        }

        // Calculate attention weights
        float attn_weight = 0.0f;
        for (int h = 0; h < num_heads; ++h) {
            float score = 0.0f;
            for (int k = 0; k < head_dim; ++k) {
                score += query[b * query_stride + i * head_dim + h * seq_len * head_dim + k] *
                         key[b * key_stride + j * head_dim + h * seq_len * head_dim + k];
            }
            attn_weight += expf(score);
        }

        // Apply attention mask
        if (attn_mask[i * seq_len + j] == 0) {
            attn_weight = 0.0f;
        }

        // Calculate output
        float output_value = 0.0f;
        for (int h = 0; h < num_heads; ++h) {
            for (int k = 0; k < head_dim; ++k) {
                output_value += value[b * value_stride + j * head_dim + h * seq_len * head_dim + k] *
                              attn_weight;
            }
        }
        output[row * output_stride + col] = output_value;

        // Store attention weights
        attn_weights[b * attn_weights_stride + i * seq_len + j] = attn_weight;
    }
}

// Forward kernel for the linear layer
__global__ void linear_forward_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int input_dim, int output_dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < output_dim) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            sum += input[row * input_dim + i] * weight[i * output_dim + col];
        }
        output[row * output_dim + col] = sum + bias[col];
    }
}

// Forward kernel for the ReLU activation
__global__ void relu_forward_kernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        input[idx] = fmaxf(input[idx], 0.0f);
    }
}

// Forward kernel for the layer normalization
__global__ void layer_norm_forward_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int seq_len, int hidden_dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size * seq_len && col < hidden_dim) {
        int b = row / seq_len;
        int i = row % seq_len;

        float sum = 0.0f;
        for (int k = 0; k < hidden_dim; ++k) {
            sum += input[b * seq_len * hidden_dim + i * hidden_dim + k];
        }
        float mean = sum / hidden_dim;
        float variance = 0.0f;
        for (int k = 0; k < hidden_dim; ++k) {
            variance += (input[b * seq_len * hidden_dim + i * hidden_dim + k] - mean) *
                        (input[b * seq_len * hidden_dim + i * hidden_dim + k] - mean);
        }
        variance /= hidden_dim;
        output[b * seq_len * hidden_dim + i * hidden_dim + col] =
            (input[b * seq_len * hidden_dim + i * hidden_dim + col] - mean) / sqrtf(variance + 1e-5) *
            weight[col] + bias[col];
    }
}

// Forward kernel for the dropout layer
__global__ void dropout_forward_kernel(
    const float* input, float* output, float dropout_prob, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if ((float)rand() / RAND_MAX < dropout_prob) {
            output[idx] = 0.0f;
        } else {
            output[idx] = input[idx];
        }
    }
}

extern "C" {

void transformer_decoder_layer(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* tgt = va_arg(args, const float*);
    int tgt_dim0 = va_arg(args, int);
    int tgt_dim1 = va_arg(args, int);
    int tgt_dim2 = va_arg(args, int);

    const float* memory = va_arg(args, const float*);
    int memory_dim0 = va_arg(args, int);
    int memory_dim1 = va_arg(args, int);
    int memory_dim2 = va_arg(args, int);

    const bool* tgt_mask = va_arg(args, const bool*);
    int tgt_mask_dim0 = va_arg(args, int);
    int tgt_mask_dim1 = va_arg(args, int);

    const bool* tgt_key_padding_mask = va_arg(args, const bool*);
    int tgt_key_padding_mask_dim0 = va_arg(args, int);
    int tgt_key_padding_mask_dim1 = va_arg(args, int);

    const bool* memory_key_padding_mask = va_arg(args, const bool*);
    int memory_key_padding_mask_dim0 = va_arg(args, int);
    int memory_key_padding_mask_dim1 = va_arg(args, int);

    const float* q_proj_weight = va_arg(args, const float*);
    int q_proj_weight_dim0 = va_arg(args, int);
    int q_proj_weight_dim1 = va_arg(args, int);

    const float* k_proj_weight = va_arg(args, const float*);
    int k_proj_weight_dim0 = va_arg(args, int);
    int k_proj_weight_dim1 = va_arg(args, int);

    const float* v_proj_weight = va_arg(args, const float*);
    int v_proj_weight_dim0 = va_arg(args, int);
    int v_proj_weight_dim1 = va_arg(args, int);

    const float* in_proj_weight = va_arg(args, const float*);
    int in_proj_weight_dim0 = va_arg(args, int);
    int in_proj_weight_dim1 = va_arg(args, int);

    const float* in_proj_bias = va_arg(args, const float*);
    int in_proj_bias_dim0 = va_arg(args, int);

    const float* linear1_weight = va_arg(args, const float*);
    int linear1_weight_dim0 = va_arg(args, int);
    int linear1_weight_dim1 = va_arg(args, int);

    const float* linear1_bias = va_arg(args, const float*);
    int linear1_bias_dim0 = va_arg(args, int);

    const float* linear2_weight = va_arg(args, const float*);
    int linear2_weight_dim0 = va_arg(args, int);
    int linear2_weight_dim1 = va_arg(args, int);

    const float* linear2_bias = va_arg(args, const float*);
    int linear2_bias_dim0 = va_arg(args, int);

    const float* dropout1_prob = va_arg(args, const float*);
    const float* dropout2_prob = va_arg(args, const float*);
    const float* dropout3_prob = va_arg(args, const float*);

    const char* activation = va_arg(args, const char*);

    const float* norm1_weight = va_arg(args, const float*);
    int norm1_weight_dim0 = va_arg(args, int);

    const float* norm1_bias = va_arg(args, const float*);
    int norm1_bias_dim0 = va_arg(args, int);

    const float* norm2_weight = va_arg(args, const float*);
    int norm2_weight_dim0 = va_arg(args, int);

    const float* norm2_bias = va_arg(args, const float*);
    int norm2_bias_dim0 = va_arg(args, int);

    float* output = va_arg(args, float*);
    int output_dim0 = va_arg(args, int);
    int output_dim1 = va_arg(args, int);
    int output_dim2 = va_arg(args, int);

    float* attn_weights = va_arg(args, float*);
    int attn_weights_dim0 = va_arg(args, int);
    int attn_weights_dim1 = va_arg(args, int);
    int attn_weights_dim2 = va_arg(args, int);

    va_end(args);

    int batch_size = tgt_dim0;
    int seq_len = tgt_dim1;
    int hidden_dim = tgt_dim2;

    int num_heads = q_proj_weight_dim0 / hidden_dim;
    int head_dim = hidden_dim / num_heads;

    // Allocate device memory
    float *d_tgt, *d_memory, *d_q_proj_weight, *d_k_proj_weight, *d_v_proj_weight, *d_in_proj_weight, *d_in_proj_bias, *d_linear1_weight, *d_linear1_bias, *d_linear2_weight, *d_linear2_bias, *d_norm1_weight, *d_norm1_bias, *d_norm2_weight, *d_norm2_bias, *d_output, *d_attn_weights;
    CUDA_CHECK(cudaMalloc(&d_tgt, batch_size * seq_len * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_memory, batch_size * seq_len * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_proj_weight, q_proj_weight_dim0 * q_proj_weight_dim1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_proj_weight, k_proj_weight_dim0 * k_proj_weight_dim1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_proj_weight, v_proj_weight_dim0 * v_proj_weight_dim1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in_proj_weight, in_proj_weight_dim0 * in_proj_weight_dim1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in_proj_bias, in_proj_bias_dim0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_linear1_weight, linear1_weight_dim0 * linear1_weight_dim1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_linear1_bias, linear1_bias_dim0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_linear2_weight, linear2_weight_dim0 * linear2_weight_dim1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_linear2_bias, linear2_bias_dim0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm1_weight, norm1_weight_dim0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm1_bias, norm1_bias_dim0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm2_weight, norm2_weight_dim0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm2_bias, norm2_bias_dim0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * seq_len * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_weights, batch_size * seq_len * seq_len * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_tgt, tgt, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_memory, memory, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_proj_weight, q_proj_weight, q_proj_weight_dim0 * q_proj_weight_dim1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_proj_weight, k_proj_weight, k_proj_weight_dim0 * k_proj_weight_dim1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_proj_weight, v_proj_weight, v_proj_weight_dim0 * v_proj_weight_dim1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_proj_weight, in_proj_weight, in_proj_weight_dim0 * in_proj_weight_dim1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_proj_bias, in_proj_bias, in_proj_bias_dim0 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_linear1_weight, linear1_weight, linear1_weight_dim0 * linear1_weight_dim1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_linear1_bias, linear1_bias, linear1_bias_dim0 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_linear2_weight, linear2_weight, linear2_weight_dim0 * linear2_weight_dim1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_linear2_bias, linear2_bias, linear2_bias_dim0 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm1_weight, norm1_weight, norm1_weight_dim0 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm1_bias, norm1_bias, norm1_bias_dim0 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm2_weight, norm2_weight, norm2_weight_dim0 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm2_bias, norm2_bias, norm2_bias_dim0 * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernels
    dim3 block_size(32, 32);
    dim3 grid_size(1, (batch_size * seq_len + block_size.y - 1) / block_size.y);

    // Linear layer
    linear_forward_kernel<<<grid_size, block_size>>>(d_tgt, d_in_proj_weight, d_in_proj_bias, d_tgt, batch_size, hidden_dim, hidden_dim);

    // ReLU activation
    relu_forward_kernel<<<grid_size, block_size>>>(d_tgt, batch_size * seq_len * hidden_dim);

    // Dropout layer
    dropout_forward_kernel<<<grid_size, block_size>>>(d_tgt, d_tgt, *dropout1_prob, batch_size * seq_len * hidden_dim);

    // Layer normalization
    layer_norm_forward_kernel<<<grid_size, block_size>>>(d_tgt, d_norm1_weight, d_norm1_bias, d_tgt, batch_size, seq_len, hidden_dim);

    // Multihead attention
    grid_size = dim3((seq_len + block_size.x - 1) / block_size.x, (batch_size * seq_len + block_size.y - 1) / block_size.y);
    multihead_attention_forward_kernel<<<grid_size, block_size>>>(d_tgt, d_memory, d_memory, d_tgt_mask, d_memory_key_padding_mask, d_q_proj_weight, d_k_proj_weight, d_v_proj_weight, d_tgt, d_attn_weights, batch_size, seq_len, num_heads, head_dim, seq_len * hidden_dim, seq_len * hidden_dim, seq_len * hidden_dim, seq_len * hidden_dim, seq_len * seq_len);

    // Dropout layer
    dropout_forward_kernel<<<grid_size, block_size>>>(d_tgt, d_tgt, *dropout3_prob, batch_size * seq_len * hidden_dim);

    // Layer normalization
    layer_norm_forward_kernel<<<grid_size, block_size>>>(d_tgt, d_norm2_weight, d_norm2_bias, d_tgt, batch_size, seq_len, hidden_dim);

    // Linear layer
    linear_forward_kernel<<<grid_size, block_size>>>(d_tgt, d_linear1_weight, d_linear1_bias, d_tgt, batch_size, hidden_dim, hidden_dim);

    // ReLU activation
    relu_forward_kernel<<<grid_size, block_size>>>(d_tgt, batch_size * seq_len * hidden_dim);

    // Dropout layer
    dropout_forward_kernel<<<grid_size, block_size>>>(d_tgt, d_tgt, *dropout1_prob, batch_size * seq_len * hidden_dim);

    // Linear layer
    linear_forward_kernel<<<grid_size, block_size>>>(d_tgt, d_linear2_weight, d_linear2_bias, d_tgt, batch_size, hidden_dim, hidden_dim);

    // Add residual
    CUDA_CHECK(cudaMemcpy(d_output, d_tgt, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt, d_output, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice));

    // Copy output data back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(attn_weights, d_attn_weights, batch_size * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_tgt));
    CUDA_CHECK(cudaFree(d_memory));
    CUDA_CHECK(cudaFree(d_q_proj_weight));
    CUDA_CHECK(cudaFree(d_k_proj_weight));
    CUDA_CHECK(cudaFree(d_v_proj_weight));
    CUDA_CHECK(cudaFree(d_in_proj_weight));
    CUDA_CHECK(cudaFree(d_in_proj_bias));
    CUDA_CHECK(cudaFree(d_linear1_weight));
    CUDA_CHECK(cudaFree(d_linear1_bias));
    CUDA_CHECK(cudaFree(d_linear2_weight));
    CUDA_CHECK(cudaFree(d_linear2_bias));
    CUDA_CHECK(cudaFree(d_norm1_weight));
    CUDA_CHECK(cudaFree(d_norm1_bias));
    CUDA_CHECK(cudaFree(d_norm2_weight));
    CUDA_CHECK(cudaFree(d_norm2_bias));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_attn_weights));
}

} // extern "C"

