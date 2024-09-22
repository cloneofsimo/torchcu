
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// Swish activation function
__device__ __forceinline__ float swish(float x) {
    return x * expf(x) / (1 + expf(x));
}

// ReLU6 activation function
__device__ __forceinline__ float relu6(float x) {
    return fminf(fmaxf(x, 0.0f), 6.0f);
}

__global__ void dynamic_positional_encoding_swish_pairwise_distance_kernel(
    const float* input_tensor, const float* embeddings, float* output,
    int batch_size, int seq_length, int embedding_dim, int vocab_size, int max_length) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < seq_length) {
        float sum = 0.0f;
        for (int i = 0; i < embedding_dim; ++i) {
            float input_value = input_tensor[(row * seq_length + col) * embedding_dim + i];
            float position_embedding = sinf(i / 10000.0f * (col + 1) / seq_length); 
            input_value += position_embedding; 
            input_value = swish(input_value);
            input_value = relu6(input_value); // Apply ReLU6 for numerical stability
            sum += (input_value - input_tensor[(row * seq_length + col) * embedding_dim + i]) * 
                   (input_value - input_tensor[(row * seq_length + col) * embedding_dim + i]);
        }
        output[(row * seq_length + col) * seq_length + col] = sum;
    }
}


__global__ void pairwise_distance_kernel(
    const float* input_tensor, float* output,
    int batch_size, int seq_length, int embedding_dim) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < seq_length) {
        float sum = 0.0f;
        for (int i = 0; i < embedding_dim; ++i) {
            sum += (input_tensor[(row * seq_length + col) * embedding_dim + i] - 
                    input_tensor[(row * seq_length + col) * embedding_dim + i]) *
                    (input_tensor[(row * seq_length + col) * embedding_dim + i] - 
                    input_tensor[(row * seq_length + col) * embedding_dim + i]);
        }
        output[(row * seq_length + col) * seq_length + col] = sum;
    }
}

extern "C" {

void dynamic_positional_encoding_swish_pairwise_distance(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int seq_length = va_arg(args, int);
    int embedding_dim = va_arg(args, int);

    // Extract embeddings tensor
    const float* embeddings = va_arg(args, const float*);
    int vocab_size = va_arg(args, int);

    // Extract max_length
    int max_length = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_embeddings, *d_output;
    cudaMalloc(&d_input, batch_size * seq_length * embedding_dim * sizeof(float));
    cudaMalloc(&d_embeddings, vocab_size * embedding_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * seq_length * seq_length * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * seq_length * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_embeddings, embeddings, vocab_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((seq_length + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    dynamic_positional_encoding_swish_pairwise_distance_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_embeddings, d_output, batch_size, seq_length, embedding_dim, vocab_size, max_length
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * seq_length * seq_length * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_embeddings);
    cudaFree(d_output);
}

}  // extern "C"

