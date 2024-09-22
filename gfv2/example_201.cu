
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// Dynamic positional encoding
__global__ void dynamic_positional_encoding_kernel(float* input, int seq_len, int hidden_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seq_len) {
        for (int j = 0; j < hidden_dim; j += 2) {
            float pos = (float)i / (10000.0f ** ((float)j / hidden_dim));
            input[i * hidden_dim + j] = sinf(pos);
            input[i * hidden_dim + j + 1] = cosf(pos);
        }
    }
}

// Window attention core kernel (fp16 and bf16)
__global__ void window_attention_core_kernel(const half* input, const int* window_mask, half* output,
                                          int batch_size, int window_len, int hidden_dim, int num_heads) {
    int b = blockIdx.z * blockDim.z + threadIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && h < num_heads && i < window_len) {
        int head_dim = hidden_dim / num_heads;
        int q_index = b * num_heads * window_len + h * window_len + i;
        int k_index = b * num_heads * window_len + h * window_len + i;
        int v_index = b * num_heads * window_len + h * window_len + i;

        float sum = 0.0f;
        for (int j = 0; j < window_len; ++j) {
            if (window_mask[i * window_len + j]) {
                __nv_bfloat16 q_val = float_to_bfloat16(__half2float(input[q_index + j * head_dim]));
                __nv_bfloat16 k_val = float_to_bfloat16(__half2float(input[k_index + j * head_dim]));
                __nv_bfloat16 v_val = float_to_bfloat16(__half2float(input[v_index + j * head_dim]));

                __nv_bfloat16 score = __hmul(q_val, k_val) / sqrtf((float)head_dim);
                __nv_bfloat16 attention = expf(__bfloat162float(score));
                sum += bfloat16_to_float(__hmul(attention, v_val));
            }
        }

        output[b * num_heads * window_len + h * window_len + i] = __float2half(sum);
    }
}

// Kernel for matrix multiplication
__global__ void matmul_kernel(const half* input, const half* weight, half* output,
                             int batch_size, int input_dim, int output_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < output_dim) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            sum += __half2float(input[row * input_dim + i]) * __half2float(weight[col * input_dim + i]);
        }
        output[row * output_dim + col] = __float2half(sum);
    }
}

extern "C" {

void window_attention_fp16_bf16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int seq_len = va_arg(args, int);
    int hidden_dim = va_arg(args, int);

    // Extract window size and num_heads
    int window_size = va_arg(args, int);
    int num_heads = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    half *d_input_fp16, *d_weight_fp16, *d_output_fp16;
    int *d_window_mask;
    cudaMalloc(&d_input, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_input_fp16, batch_size * seq_len * hidden_dim * sizeof(half));
    cudaMalloc(&d_weight_fp16, hidden_dim * hidden_dim * sizeof(half));
    cudaMalloc(&d_output_fp16, batch_size * seq_len * hidden_dim * sizeof(half));
    cudaMalloc(&d_window_mask, seq_len * seq_len * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate dynamic positional encoding
    dynamic_positional_encoding_kernel<<<(seq_len + 31) / 32, 32>>>(d_input, seq_len, hidden_dim);

    // Convert input to fp16
    cudaMemcpy(d_input_fp16, d_input, batch_size * seq_len * hidden_dim * sizeof(half), cudaMemcpyDeviceToDevice);

    // Calculate window masks
    int* window_mask = new int[seq_len * seq_len];
    for (int i = 0; i < seq_len; ++i) {
        int start = max(0, i - window_size);
        int end = min(seq_len, i + window_size + 1);
        for (int j = start; j < end; ++j) {
            window_mask[i * seq_len + j] = 1;
        }
    }
    cudaMemcpy(d_window_mask, window_mask, seq_len * seq_len * sizeof(int), cudaMemcpyHostToDevice);
    delete[] window_mask;

    // Perform window attention
    int window_len = window_size;
    for (int i = 0; i < seq_len; i += window_len) {
        // Launch window attention core kernel
        dim3 threadsPerBlock(32, 8, 1);
        dim3 numBlocks((window_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (num_heads + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);
        window_attention_core_kernel<<<numBlocks, threadsPerBlock>>>(
            d_input_fp16 + i * batch_size * hidden_dim, d_window_mask + i * seq_len + i,
            d_output_fp16 + i * batch_size * hidden_dim, batch_size, window_len, hidden_dim, num_heads
        );

        // Project output
        dim3 threadsPerBlock2(32, 8);
        dim3 numBlocks2((hidden_dim + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
                        (batch_size * window_len + threadsPerBlock2.y - 1) / threadsPerBlock2.y);
        matmul_kernel<<<numBlocks2, threadsPerBlock2>>>(
            d_output_fp16 + i * batch_size * hidden_dim, d_weight_fp16,
            d_output_fp16 + i * batch_size * hidden_dim, batch_size, window_len, hidden_dim
        );
    }

    // Copy result back to host
    cudaMemcpy(output, d_output_fp16, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_input_fp16);
    cudaFree(d_weight_fp16);
    cudaFree(d_output_fp16);
    cudaFree(d_window_mask);
}

} // extern "C"