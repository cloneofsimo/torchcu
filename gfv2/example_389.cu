
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

#define MISH_THRESHOLD 20.0f
#define MISH_SCALE 0.44715f

// Helper function for Mish activation
__device__ __forceinline__ float mish_activation(float x) {
    float x_tanh = tanhf(x * MISH_SCALE);
    return x * x_tanh * x_tanh;
}

// CUDA kernel for Mish activation, global attention, and multiplication
__global__ void mish_global_attention_mul_kernel(const float* input_tensor, const float* attention_weights, float* output,
                                             int batch_size, int seq_len, int hidden_dim, float scale) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && seq_idx < seq_len && hidden_idx < hidden_dim) {
        int input_index = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + hidden_idx;
        int attention_index = batch_idx * seq_len + seq_idx;

        float mish_value = mish_activation(input_tensor[input_index]);
        float attention_value = attention_weights[attention_index];

        output[input_index] = mish_value * attention_value * scale;
    }
}

extern "C" {

void mish_global_attention_mul(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract attention weights tensor
    const float* attention_weights = va_arg(args, const float*);
    int attention_weights_dim0 = va_arg(args, int);
    int attention_weights_dim1 = va_arg(args, int);

    // Extract scale
    float scale = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int seq_len = input_tensor_dim1;
    int hidden_dim = input_tensor_dim2;

    // Allocate device memory
    float *d_input, *d_attention_weights, *d_output;
    cudaMalloc(&d_input, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_attention_weights, batch_size * seq_len * sizeof(float));
    cudaMalloc(&d_output, batch_size * seq_len * hidden_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attention_weights, attention_weights, batch_size * seq_len * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((hidden_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    mish_global_attention_mul_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_attention_weights, d_output, batch_size, seq_len, hidden_dim, scale
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_attention_weights);
    cudaFree(d_output);
}

}  // extern "C"

