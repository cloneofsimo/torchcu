
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for causal attention with bfloat16
__global__ void causal_attention_kernel_bf16(const float* query, const float* key, const float* value, float* output, 
                                           int batch_size, int seq_len, int hidden_dim) {
    int b = blockIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < batch_size && i < seq_len && j < seq_len) {
        float sum = 0.0f;
        float scale = 1.0f / sqrtf(hidden_dim); // Normalize scores
        for (int k = 0; k < hidden_dim; k++) {
            __nv_bfloat16 q = float_to_bfloat16(query[b * seq_len * hidden_dim + i * hidden_dim + k]);
            __nv_bfloat16 k_t = float_to_bfloat16(key[b * seq_len * hidden_dim + j * hidden_dim + k]);
            sum += bfloat16_to_float(__hmul(q, k_t)) * scale;
        }

        // Apply causal mask
        if (i < j) {
            sum = -INFINITY;
        }

        // Softmax normalization (local softmax within each sequence)
        float max_val = -INFINITY;
        for (int j_prime = 0; j_prime < seq_len; j_prime++) {
            if (i < j_prime) {
                float tmp = sum;
                for (int k = 0; k < hidden_dim; k++) {
                    __nv_bfloat16 q = float_to_bfloat16(query[b * seq_len * hidden_dim + i * hidden_dim + k]);
                    __nv_bfloat16 k_t = float_to_bfloat16(key[b * seq_len * hidden_dim + j_prime * hidden_dim + k]);
                    tmp += bfloat16_to_float(__hmul(q, k_t)) * scale;
                }
                max_val = fmaxf(max_val, tmp);
            }
        }

        float exp_sum = 0.0f;
        for (int j_prime = 0; j_prime < seq_len; j_prime++) {
            if (i < j_prime) {
                float tmp = sum;
                for (int k = 0; k < hidden_dim; k++) {
                    __nv_bfloat16 q = float_to_bfloat16(query[b * seq_len * hidden_dim + i * hidden_dim + k]);
                    __nv_bfloat16 k_t = float_to_bfloat16(key[b * seq_len * hidden_dim + j_prime * hidden_dim + k]);
                    tmp += bfloat16_to_float(__hmul(q, k_t)) * scale;
                }
                exp_sum += expf(tmp - max_val);
            }
        }

        // Weighted sum
        float attention_weight = expf(sum - max_val) / exp_sum;
        for (int k = 0; k < hidden_dim; k++) {
            __nv_bfloat16 v = float_to_bfloat16(value[b * seq_len * hidden_dim + j * hidden_dim + k]);
            output[b * seq_len * hidden_dim + i * hidden_dim + k] += attention_weight * bfloat16_to_float(v);
        }
    }
}

extern "C" {

void causal_attention_bf16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract tensors
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

    float* output = va_arg(args, float*);

    va_end(args);

    // Check if input dimensions are consistent
    if (query_dim0 != key_dim0 || query_dim0 != value_dim0 ||
        query_dim1 != key_dim1 || query_dim1 != value_dim1 ||
        query_dim2 != key_dim2 || query_dim2 != value_dim2) {
        // Handle dimension mismatch error
        return;
    }

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
    dim3 threadsPerBlock(1, 16, 16);
    dim3 numBlocks(batch_size, (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (seq_len + threadsPerBlock.z - 1) / threadsPerBlock.z);

    causal_attention_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_query, d_key, d_value, d_output, batch_size, seq_len, hidden_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
}

} // extern "C"
