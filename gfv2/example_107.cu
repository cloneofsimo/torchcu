
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

__device__ __forceinline__ half float_to_half(float f) {
    return __float2half(f);
}

__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

__global__ void masked_attention_kernel(const float* query, const float* key, const float* value,
                                         const float* mask, float* output,
                                         int batch_size, int query_len, int key_len, int d_model) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && q < query_len) {
        float sum = 0.0f;
        for (int k = 0; k < key_len; ++k) {
            float q_val = query[b * query_len * d_model + q * d_model];
            float k_val = key[b * key_len * d_model + k * d_model];
            float score = q_val * k_val / sqrtf((float)d_model);
            score += mask[b * query_len * key_len + q * key_len + k];

            // Apply softmax in fp16
            half score_half = __float2half(score);
            half exp_half = __expf(score_half);
            sum += half_to_float(__hmul(exp_half, value[b * key_len * d_model + k * d_model]));
        }
        output[b * query_len * d_model + q * d_model] = sum;
    }
}

extern "C" {

void masked_attention_forward_fp16(int num_args, ...) {
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

    const float* mask = va_arg(args, const float*);
    int mask_dim0 = va_arg(args, int);
    int mask_dim1 = va_arg(args, int);
    int mask_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = query_dim0;
    int query_len = query_dim1;
    int key_len = key_dim1;
    int d_model = query_dim2;

    // Allocate device memory
    float *d_query, *d_key, *d_value, *d_mask, *d_output;
    cudaMalloc(&d_query, batch_size * query_len * d_model * sizeof(float));
    cudaMalloc(&d_key, batch_size * key_len * d_model * sizeof(float));
    cudaMalloc(&d_value, batch_size * key_len * d_model * sizeof(float));
    cudaMalloc(&d_mask, batch_size * query_len * key_len * sizeof(float));
    cudaMalloc(&d_output, batch_size * query_len * d_model * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_query, query, batch_size * query_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key, batch_size * key_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, batch_size * key_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, batch_size * query_len * key_len * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (query_len + threadsPerBlock.y - 1) / threadsPerBlock.y);

    masked_attention_kernel<<<numBlocks, threadsPerBlock>>>(d_query, d_key, d_value, d_mask, d_output,
                                                        batch_size, query_len, key_len, d_model);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * query_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_mask);
    cudaFree(d_output);
}

}  // extern "C"
