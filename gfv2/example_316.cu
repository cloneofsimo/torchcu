
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for windowed attention with bfloat16 and gradient clipping
__global__ void window_attention_kernel_bf16(const float* q, const float* k, const float* v,
                                             const bool* attn_mask, const float* relative_position_bias,
                                             float* output, int B, int N, int head_dim, int window_size,
                                             float gradient_clip_value) {
    int batch_idx = blockIdx.z;
    int window_idx = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = threadIdx.y;

    int seq_idx = window_idx * window_size + row;
    int global_col = window_idx * window_size + col;

    if (seq_idx < N && global_col < N && batch_idx < B) {
        __nv_bfloat16 sum = 0.0f;
        for (int i = 0; i < window_size; ++i) {
            int global_row = window_idx * window_size + i;
            if (attn_mask[batch_idx * N * N + seq_idx * N + global_row]) {
                __nv_bfloat16 q_val = float_to_bfloat16(q[batch_idx * N * head_dim + seq_idx * head_dim + i]);
                __nv_bfloat16 k_val = float_to_bfloat16(k[batch_idx * N * head_dim + global_row * head_dim + col]);
                __nv_bfloat16 relative_bias = float_to_bfloat16(relative_position_bias[(row - col + window_size - 1) * (2 * window_size - 1) + (i - col + window_size - 1)]);
                sum += bfloat16_to_float(__hmul(q_val, k_val)) / sqrtf(head_dim) + bfloat16_to_float(relative_bias);
            }
        }
        sum = expf(sum);  // Softmax calculation

        // Calculate the sum of exp() over all columns in the window
        __shared__ float shared_sum[32];
        if (threadIdx.x == 0) {
            shared_sum[threadIdx.y] = sum;
        }
        __syncthreads();

        // Reduce the shared sum to get the denominator for softmax
        for (int s = 16; s > 0; s >>= 1) {
            if (threadIdx.y < s) {
                shared_sum[threadIdx.y] += shared_sum[threadIdx.y + s];
            }
            __syncthreads();
        }
        float denominator = shared_sum[0];

        // Calculate the final attention weight
        sum /= denominator;

        // Multiply attention weight with value tensor
        output[batch_idx * N * head_dim + seq_idx * head_dim + col] =
            sum * v[batch_idx * N * head_dim + global_col * head_dim + col];
    }
}

extern "C" {

void window_attention_with_bfloat16_and_gradient_clipping(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input arguments
    const float* q = va_arg(args, const float*);
    int q_dim0 = va_arg(args, int);
    int q_dim1 = va_arg(args, int);
    int q_dim2 = va_arg(args, int);

    const float* k = va_arg(args, const float*);
    int k_dim0 = va_arg(args, int);
    int k_dim1 = va_arg(args, int);
    int k_dim2 = va_arg(args, int);

    const float* v = va_arg(args, const float*);
    int v_dim0 = va_arg(args, int);
    int v_dim1 = va_arg(args, int);
    int v_dim2 = va_arg(args, int);

    const bool* attn_mask = va_arg(args, const bool*);
    int attn_mask_dim0 = va_arg(args, int);
    int attn_mask_dim1 = va_arg(args, int);
    int attn_mask_dim2 = va_arg(args, int);

    int window_size = va_arg(args, int);
    int head_dim = va_arg(args, int);

    const float* relative_position_bias = va_arg(args, const float*);
    int relative_position_bias_dim0 = va_arg(args, int);
    int relative_position_bias_dim1 = va_arg(args, int);

    float gradient_clip_value = va_arg(args, double);

    float* output = va_arg(args, float*);

    va_end(args);

    // Check input dimensions and ensure consistency
    if (q_dim0 != k_dim0 || q_dim0 != v_dim0 || q_dim0 != attn_mask_dim0) {
        printf("Error: Batch size mismatch between inputs.\n");
        return;
    }
    if (q_dim1 != k_dim1 || q_dim1 != v_dim1 || q_dim1 != attn_mask_dim1) {
        printf("Error: Sequence length mismatch between inputs.\n");
        return;
    }
    if (q_dim2 != k_dim2 || q_dim2 != v_dim2 || q_dim2 != head_dim) {
        printf("Error: Head dimension mismatch between inputs.\n");
        return;
    }

    int B = q_dim0;
    int N = q_dim1;

    // Allocate device memory for input tensors
    float *d_q, *d_k, *d_v, *d_relative_position_bias;
    bool *d_attn_mask;
    cudaMalloc(&d_q, B * N * head_dim * sizeof(float));
    cudaMalloc(&d_k, B * N * head_dim * sizeof(float));
    cudaMalloc(&d_v, B * N * head_dim * sizeof(float));
    cudaMalloc(&d_relative_position_bias,
               relative_position_bias_dim0 * relative_position_bias_dim1 * sizeof(float));
    cudaMalloc(&d_attn_mask, B * N * N * sizeof(bool));

    // Allocate device memory for output tensor
    cudaMalloc(&d_output, B * N * head_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_q, q, B * N * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, B * N * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, B * N * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_relative_position_bias,
               relative_position_bias,
               relative_position_bias_dim0 * relative_position_bias_dim1 * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_attn_mask, attn_mask, B * N * N * sizeof(bool), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((window_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (window_size + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   B);

    window_attention_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_q, d_k, d_v, d_attn_mask, d_relative_position_bias, d_output, B, N, head_dim,
        window_size, gradient_clip_value);

    // Copy result back to host
    cudaMemcpy(output, d_output, B * N * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_relative_position_bias);
    cudaFree(d_attn_mask);
    cudaFree(d_output);
}

}  // extern "C"
