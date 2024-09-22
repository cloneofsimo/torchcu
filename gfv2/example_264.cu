
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256

__global__ void max_pool_scatter_add_fp16_kernel(const half* input, const int* indices, const int* lengths, 
                                                half* output, int batch_size, int seq_len, int max_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch_size) {
        int start = 0;
        int end = lengths[tid];
        for (int i = start; i < end; i++) {
            int idx = tid * seq_len + i;
            if (input[idx] > output[indices[tid] + i]) {
                output[indices[tid] + i] = input[idx];
            }
        }
    }
}

extern "C" {

void max_pool_scatter_add_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);

    // Extract indices tensor
    const int* indices = va_arg(args, const int*);
    int indices_dim0 = va_arg(args, int);

    // Extract lengths tensor
    const int* lengths = va_arg(args, const int*);
    int lengths_dim0 = va_arg(args, int);

    // Extract output tensor (pre-allocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    half* d_input;
    int* d_indices;
    int* d_lengths;
    half* d_output;
    cudaMalloc(&d_input, input_dim0 * input_dim1 * sizeof(half));
    cudaMalloc(&d_indices, indices_dim0 * sizeof(int));
    cudaMalloc(&d_lengths, lengths_dim0 * sizeof(int));
    cudaMalloc(&d_output, lengths_dim0 * input_dim1 * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input, input, input_dim0 * input_dim1 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices, indices_dim0 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths, lengths_dim0 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int max_len = lengths_dim0 * input_dim1;
    max_pool_scatter_add_fp16_kernel<<<(input_dim0 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_input, d_indices, d_lengths, d_output, input_dim0, input_dim1, max_len);

    // Copy result back to host
    cudaMemcpy(output, d_output, max_len * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_indices);
    cudaFree(d_lengths);
    cudaFree(d_output);
}

} // extern "C"
