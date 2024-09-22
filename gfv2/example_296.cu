
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __half
__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half_rn(f);
}

// CUDA kernel for calculating Hamming distance
__global__ void pairwise_hamming_distance_kernel(const int8_t* x, const int8_t* y, int* distance, int seq_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < seq_len && j < seq_len) {
        distance[i * seq_len + j] = __popc(x[i] ^ y[j]);
    }
}

// CUDA kernel for relative positional encoding
__global__ void relative_positional_encoding_kernel(const int* distance, __half* relative_position_embeddings, 
                                                    int batch_size, int seq_len, int max_relative_positions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < seq_len && j < seq_len) {
        int dist = distance[i * seq_len + j];
        int index = dist + max_relative_positions;
        relative_position_embeddings[i * seq_len + j] = float_to_half(index < 2 * max_relative_positions + 1 ? 1.0f : 0.0f);
    }
}

extern "C" {

void relative_positional_encoding_int8(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const int8_t* input_tensor = va_arg(args, const int8_t*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract seq_len
    int seq_len = va_arg(args, int);

    // Extract max_relative_positions
    int max_relative_positions = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    __half* output = va_arg(args, __half*);

    va_end(args);

    // Allocate device memory
    int* d_distance;
    cudaMalloc(&d_distance, seq_len * seq_len * sizeof(int));
    cudaMalloc(&output, seq_len * seq_len * sizeof(__half));

    // Calculate pairwise Hamming distances on the device
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y);
    pairwise_hamming_distance_kernel<<<numBlocks, threadsPerBlock>>>(input_tensor, input_tensor, d_distance, seq_len);

    // Compute relative positional encodings on the device
    relative_positional_encoding_kernel<<<numBlocks, threadsPerBlock>>>(d_distance, output, 1, seq_len, max_relative_positions);

    // Free device memory
    cudaFree(d_distance);
}

}  // extern "C"
