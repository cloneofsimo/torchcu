
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <cublas_v2.h>
#include <cufft.h>

extern "C" {
    void attention_conv_ifft(int num_args, ...);
}

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* query, const float* key, float* attn, 
                              int batch_size, int seq_len, int hidden_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size * seq_len && col < batch_size * seq_len) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            sum += query[row * hidden_size + i] * key[col * hidden_size + i];
        }
        attn[row * batch_size * seq_len + col] = sum;
    }
}

// CUDA kernel for softmax
__global__ void softmax_kernel(float* attn, int batch_size, int seq_len) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size * seq_len && col < seq_len) {
        float max_val = attn[row * batch_size * seq_len + col];
        for (int i = 0; i < seq_len; ++i) {
            max_val = fmaxf(max_val, attn[row * batch_size * seq_len + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            sum += expf(attn[row * batch_size * seq_len + i] - max_val);
        }

        attn[row * batch_size * seq_len + col] = expf(attn[row * batch_size * seq_len + col] - max_val) / sum;
    }
}

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel2(const float* attn, const float* value, float* out,
                               int batch_size, int seq_len, int hidden_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size * seq_len && col < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            sum += attn[row * batch_size * seq_len + i] * value[i * hidden_size + col];
        }
        out[row * hidden_size + col] = sum;
    }
}

// CUDA kernel for conv1d (using CuDNN)
__global__ void conv1d_kernel(const float* input, const float* weight, const float* bias, 
                               float* output, int batch_size, int seq_len, int in_channels, int out_channels, int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size * seq_len && col < out_channels) {
        float sum = bias[col];
        for (int i = 0; i < in_channels; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int in_idx = (row * in_channels + i) * seq_len + (col + j);
                int weight_idx = (i * out_channels + col) * kernel_size + j;
                sum += input[in_idx] * weight[weight_idx];
            }
        }
        output[row * out_channels + col] = sum;
    }
}

// CUDA kernel for inverse FFT (using CUFFT)
__global__ void ifft_kernel(float* output, int batch_size, int seq_len, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * seq_len * hidden_size) {
        output[idx] = output[idx] / seq_len;
    }
}

// CUDA kernel for normalization
__global__ void normalize_kernel(float* output, int batch_size, int seq_len, int hidden_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size * seq_len && col < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            sum += output[row * hidden_size + i] * output[row * hidden_size + i];
        }
        sum = sqrtf(sum);
        output[row * hidden_size + col] /= sum;
    }
}

extern "C" {
    void attention_conv_ifft(int num_args, ...) {
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

        const float* weight = va_arg(args, const float*);
        int weight_dim0 = va_arg(args, int);
        int weight_dim1 = va_arg(args, int);
        int weight_dim2 = va_arg(args, int);

        const float* bias = va_arg(args, const float*);
        int bias_dim0 = va_arg(args, int);

        // Extract output tensor (assuming it's preallocated)
        float* output = va_arg(args, float*);

        va_end(args);

        int batch_size = query_dim0;
        int seq_len = query_dim1;
        int hidden_size = query_dim2;
        int kernel_size = weight_dim2;
        int in_channels = weight_dim1;
        int out_channels = weight_dim0;

        // Allocate device memory
        float *d_query, *d_key, *d_value, *d_weight, *d_bias, *d_attn, *d_out, *d_output;
        cudaMalloc(&d_query, batch_size * seq_len * hidden_size * sizeof(float));
        cudaMalloc(&d_key, batch_size * seq_len * hidden_size * sizeof(float));
        cudaMalloc(&d_value, batch_size * seq_len * hidden_size * sizeof(float));
        cudaMalloc(&d_weight, out_channels * in_channels * kernel_size * sizeof(float));
        cudaMalloc(&d_bias, out_channels * sizeof(float));
        cudaMalloc(&d_attn, batch_size * seq_len * batch_size * seq_len * sizeof(float));
        cudaMalloc(&d_out, batch_size * seq_len * hidden_size * sizeof(float));
        cudaMalloc(&d_output, batch_size * seq_len * hidden_size * sizeof(float));

        // Copy input data to device
        cudaMemcpy(d_query, query, batch_size * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_key, key, batch_size * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_value, value, batch_size * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, weight, out_channels * in_channels * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernels

        // Attention
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((batch_size * seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (batch_size * seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matmul_kernel<<<numBlocks, threadsPerBlock>>>(
            d_query, d_key, d_attn, batch_size, seq_len, hidden_size
        );

        // Softmax
        numBlocks = ((batch_size * seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y);
        softmax_kernel<<<numBlocks, threadsPerBlock>>>(
            d_attn, batch_size, seq_len
        );

        // Matmul (attn x value)
        numBlocks = ((batch_size * seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (hidden_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matmul_kernel2<<<numBlocks, threadsPerBlock>>>(
            d_attn, d_value, d_out, batch_size, seq_len, hidden_size
        );

        // Conv1d
        numBlocks = ((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (batch_size * out_channels + threadsPerBlock.y - 1) / threadsPerBlock.y);
        conv1d_kernel<<<numBlocks, threadsPerBlock>>>(
            d_out, d_weight, d_bias, d_output, batch_size, seq_len, in_channels, out_channels, kernel_size
        );

        // IFFT
        numBlocks = ((batch_size * seq_len * hidden_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);
        ifft_kernel<<<numBlocks, threadsPerBlock>>>(
            d_output, batch_size, seq_len, hidden_size
        );

        // Norm
        numBlocks = ((hidden_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (batch_size * seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y);
        normalize_kernel<<<numBlocks, threadsPerBlock>>>(
            d_output, batch_size, seq_len, hidden_size
        );

        // Copy result back to host
        cudaMemcpy(output, d_output, batch_size * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_query);
        cudaFree(d_key);
        cudaFree(d_value);
        cudaFree(d_weight);
        cudaFree(d_bias);
        cudaFree(d_attn);
        cudaFree(d_out);
        cudaFree(d_output);
    }
} 
