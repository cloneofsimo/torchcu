
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Helper function to compute softmax along a dimension
__device__ __forceinline__ float softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        max_val = fmaxf(max_val, x[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }

    return max_val;
}

// CUDA kernel for local attention
__global__ void local_attention_kernel(float* input, float* output, int B, int T, int C, int window_size, int causal) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < B && t < T) {
        // Calculate start and end indices for local window
        int start_idx = t - window_size + 1;
        int end_idx = t + window_size;

        // Apply causal masking
        if (causal) {
            end_idx = min(end_idx, t + window_size);
        }

        // Clamp indices to valid range
        start_idx = max(start_idx, 0);
        end_idx = min(end_idx, T);

        int window_size_eff = end_idx - start_idx;

        // Calculate offsets for local window
        int input_offset = b * T * C + t * C;
        int output_offset = b * T * C + t * C;

        // Compute Q, K, V for local window
        float* q = input + input_offset;
        float* k = input + (b * T * C + start_idx * C);
        float* v = input + (b * T * C + start_idx * C);

        // Calculate attention weights
        float attn[window_size_eff * C];
        for (int i = 0; i < window_size_eff; ++i) {
            for (int j = 0; j < C; ++j) {
                attn[i * C + j] = q[j] * k[i * C + j];
            }
        }
        for (int i = 0; i < window_size_eff * C; ++i) {
            attn[i] /= sqrtf(C);
        }

        // Apply causal masking if required
        if (causal) {
            for (int i = 0; i < window_size_eff; ++i) {
                for (int j = 0; j < i; ++j) {
                    attn[i * C + j] = -INFINITY;
                }
            }
        }

        // Compute softmax
        float max_val = softmax(attn, window_size_eff * C);

        // Apply attention
        for (int i = 0; i < C; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < window_size_eff; ++j) {
                sum += attn[j * C + i] * v[j * C + i];
            }
            output[output_offset + i] = sum;
        }
    }
}

extern "C" {

void local_attention_forward(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int B = va_arg(args, int);
    int T = va_arg(args, int);
    int C = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    // Extract window size and causal flag
    int window_size = va_arg(args, int);
    int causal = va_arg(args, int);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, B * T * C * sizeof(float));
    cudaMalloc(&d_output, B * T * C * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, B * T * C * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((B + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (T + threadsPerBlock.y - 1) / threadsPerBlock.y);

    local_attention_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, B, T, C, window_size, causal
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
