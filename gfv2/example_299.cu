
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

#define BLOCK_SIZE 32

// CUDA kernel for TransformerEncoder layer
__global__ void transformer_encoder_layer_kernel(const float* src, float* dst, const float* weight, int batch_size, int seq_len, int d_model, int nhead, int dim_feedforward, float dropout) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < seq_len) {
        // Calculate index for the current element
        int idx = row * seq_len + col;

        // Apply TransformerEncoder layer
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            float val = src[row * seq_len + i];
            sum += weight[col * seq_len + i] * val;
        }
        dst[idx] = sum;
    }
}

// CUDA kernel for diagonal matrix multiplication
__global__ void diag_mul_kernel(const float* input, const float* weight, float* output, int batch_size, int seq_len, int d_model) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < seq_len) {
        int idx = row * seq_len + col;
        output[idx] = input[idx] * weight[col];
    }
}

// CUDA kernel for backward pass
__global__ void backward_kernel(const float* grad_output, const float* weight, float* grad_input, int batch_size, int seq_len, int d_model) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < seq_len) {
        int idx = row * seq_len + col;
        grad_input[idx] += grad_output[idx] * weight[col];
    }
}

extern "C" {
    void transformer_encoder_diag_backward(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract input tensor
        const float* input_tensor = va_arg(args, const float*);
        int input_tensor_dim0 = va_arg(args, int);
        int input_tensor_dim1 = va_arg(args, int);
        int input_tensor_dim2 = va_arg(args, int);

        // Extract weight tensor
        const float* weight = va_arg(args, const float*);
        int weight_dim0 = va_arg(args, int);

        // Extract output tensor (assuming it's preallocated)
        float* output = va_arg(args, float*);

        va_end(args);

        int batch_size = input_tensor_dim0;
        int seq_len = input_tensor_dim1;
        int d_model = input_tensor_dim2;
        int nhead = 4;
        int dim_feedforward = 2048;
        float dropout = 0.1;

        // Allocate device memory
        float* d_input, *d_weight, *d_output, *d_grad_input;
        cudaMalloc(&d_input, batch_size * seq_len * d_model * sizeof(float));
        cudaMalloc(&d_weight, d_model * sizeof(float));
        cudaMalloc(&d_output, batch_size * seq_len * d_model * sizeof(float));
        cudaMalloc(&d_grad_input, batch_size * seq_len * d_model * sizeof(float));

        // Copy input data to device
        cudaMemcpy(d_input, input_tensor, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, weight, d_model * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel for TransformerEncoder layer
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        transformer_encoder_layer_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_weight, batch_size, seq_len, d_model, nhead, dim_feedforward, dropout);

        // Launch kernel for diagonal matrix multiplication
        diag_mul_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_weight, d_output, batch_size, seq_len, d_model);

        // Launch kernel for backward pass
        cudaMemset(d_grad_input, 0, batch_size * seq_len * d_model * sizeof(float));
        backward_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_weight, d_grad_input, batch_size, seq_len, d_model);

        // Copy result back to host
        cudaMemcpy(output, d_grad_input, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_weight);
        cudaFree(d_output);
        cudaFree(d_grad_input);
    }
}
