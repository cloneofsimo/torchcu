
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

// CUDA kernel for window attention using bfloat16
__global__ void window_attention_kernel_bf16(const float* q, const float* k, const float* v, const bool* mask,
                                            float* output, int batch_size, int num_heads, int head_dim, int window_size) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int head_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int window_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && head_idx < num_heads && window_idx < window_size) {
        int q_offset = (batch_idx * num_heads * window_size + head_idx * window_size + window_idx) * head_dim;
        int k_offset = q_offset;  // k and q have the same layout
        int v_offset = q_offset;  // v and q have the same layout

        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(q[q_offset + i]);
            __nv_bfloat16 b = float_to_bfloat16(k[k_offset + i]);
            sum += bfloat16_to_float(__hmul(a, b));
        }

        float scale = 1.0f / sqrtf(head_dim);
        sum *= scale;

        // Apply mask (if provided)
        int mask_idx = batch_idx * num_heads * window_size + head_idx * window_size + window_idx;
        if (mask != nullptr && mask[mask_idx]) {
            sum = -INFINITY;
        }

        // Softmax
        float max_val = sum;
        for (int i = 1; i < window_size; ++i) {
            int other_idx = q_offset + i * head_dim;
            __nv_bfloat16 a = float_to_bfloat16(q[other_idx]);
            __nv_bfloat16 b = float_to_bfloat16(k[other_idx]);
            float other_val = bfloat16_to_float(__hmul(a, b));
            other_val *= scale;
            if (mask != nullptr && mask[mask_idx + i]) {
                other_val = -INFINITY;
            }
            max_val = fmaxf(max_val, other_val);
        }
        sum = expf(sum - max_val);
        float total_sum = sum;
        for (int i = 1; i < window_size; ++i) {
            int other_idx = q_offset + i * head_dim;
            __nv_bfloat16 a = float_to_bfloat16(q[other_idx]);
            __nv_bfloat16 b = float_to_bfloat16(k[other_idx]);
            float other_val = bfloat16_to_float(__hmul(a, b));
            other_val *= scale;
            if (mask != nullptr && mask[mask_idx + i]) {
                other_val = -INFINITY;
            }
            other_val = expf(other_val - max_val);
            total_sum += other_val;
        }
        sum /= total_sum;

        // Weighted sum of values
        sum *= float_to_bfloat16(v[v_offset]);  // First value
        for (int i = 1; i < head_dim; ++i) {
            int other_idx = v_offset + i;
            __nv_bfloat16 a = float_to_bfloat16(q[q_offset + i]);
            __nv_bfloat16 b = float_to_bfloat16(k[q_offset + i]);
            float other_val = bfloat16_to_float(__hmul(a, b));
            other_val *= scale;
            if (mask != nullptr && mask[mask_idx + i]) {
                other_val = -INFINITY;
            }
            other_val = expf(other_val - max_val);
            other_val /= total_sum;
            sum += other_val * float_to_bfloat16(v[other_idx]);
        }

        output[q_offset] = bfloat16_to_float(sum);
    }
}

extern "C" {

void window_attention_bf16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
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

    const bool* mask = va_arg(args, const bool*);
    int mask_dim0 = va_arg(args, int);
    int mask_dim1 = va_arg(args, int);
    int mask_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = q_dim0;
    int num_heads = q_dim1;
    int head_dim = q_dim2;
    int window_size = mask_dim2;

    // Allocate device memory
    float *d_q, *d_k, *d_v, *d_output;
    bool *d_mask = nullptr;

    cudaMalloc(&d_q, batch_size * num_heads * window_size * head_dim * sizeof(float));
    cudaMalloc(&d_k, batch_size * num_heads * window_size * head_dim * sizeof(float));
    cudaMalloc(&d_v, batch_size * num_heads * window_size * head_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * num_heads * window_size * head_dim * sizeof(float));

    if (mask != nullptr) {
        cudaMalloc(&d_mask, batch_size * num_heads * window_size * sizeof(bool));
        cudaMemcpy(d_mask, mask, batch_size * num_heads * window_size * sizeof(bool), cudaMemcpyHostToDevice);
    }

    // Copy input data to device
    cudaMemcpy(d_q, q, batch_size * num_heads * window_size * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, batch_size * num_heads * window_size * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, batch_size * num_heads * window_size * head_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((window_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_heads + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    window_attention_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_q, d_k, d_v, d_mask, d_output, batch_size, num_heads, head_dim, window_size
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * num_heads * window_size * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_output);
    if (d_mask != nullptr) {
        cudaFree(d_mask);
    }
}

}  // extern "C"
