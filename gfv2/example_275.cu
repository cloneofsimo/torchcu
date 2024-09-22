
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for attention
__global__ void attention_kernel(const float* input, float* output, int batch_size, int seq_len, int embed_dim, int num_heads) {
    int head_dim = embed_dim / num_heads;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < seq_len) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            float q = input[row * embed_dim + (col * num_heads + i)];
            float k = input[col * embed_dim + (i * num_heads + row)];

            half q_h = float_to_half(q);
            half k_h = float_to_half(k);

            half attn = __hmul(q_h, k_h) / sqrtf(head_dim); // Half-precision dot product and scaling

            sum += half_to_float(attn);
        }
        output[row * embed_dim + col] = sum;
    }
}

// CUDA kernel for squaring the result
__global__ void square_kernel(const float* input, int8_t* output, int batch_size, int seq_len, int embed_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < embed_dim) {
        float val = input[row * embed_dim + col];
        output[row * embed_dim + col] = static_cast<int8_t>(val * val);
    }
}

extern "C" {

void attention_square(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int seq_len = va_arg(args, int);
    int embed_dim = va_arg(args, int);

    int8_t* output = va_arg(args, int8_t*);

    va_end(args);

    // Allocate device memory
    float* d_input, *d_output;
    cudaMalloc(&d_input, batch_size * seq_len * embed_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * seq_len * embed_dim * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, input, batch_size * seq_len * embed_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch attention kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y);

    attention_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, batch_size, seq_len, embed_dim, 4);

    // Launch squaring kernel
    numBlocks = ((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (embed_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);
    square_kernel<<<numBlocks, threadsPerBlock>>>(d_output, output, batch_size, seq_len, embed_dim);

    // Copy output to host
    cudaMemcpy(output, d_output, batch_size * seq_len * embed_dim * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

} // extern "C"
