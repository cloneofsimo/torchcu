
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// CUDA kernel for pairwise Hamming distance computation
__global__ void pairwise_hamming_distance_kernel(const half* input1, const half* input2, float* output, 
                                              int batch_size, int num_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < batch_size && j < batch_size) {
        float distance = 0.0f;
        for (int k = 0; k < num_features; ++k) {
            distance += __hadd(input1[i * num_features + k], input2[j * num_features + k]);
        }
        output[i * batch_size + j] = distance;
    }
}

extern "C" {
    void torch_function(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract input tensors
        const half* input1 = va_arg(args, const half*);
        int input1_dim0 = va_arg(args, int);
        int input1_dim1 = va_arg(args, int);

        const half* input2 = va_arg(args, const half*);
        int input2_dim0 = va_arg(args, int);
        int input2_dim1 = va_arg(args, int);

        // Extract output tensor
        float* output = va_arg(args, float*);

        va_end(args);

        int batch_size = input1_dim0;
        int num_features = input1_dim1;

        // Allocate device memory
        half *d_input1, *d_input2;
        float *d_output;
        cudaMalloc(&d_input1, batch_size * num_features * sizeof(half));
        cudaMalloc(&d_input2, batch_size * num_features * sizeof(half));
        cudaMalloc(&d_output, batch_size * batch_size * sizeof(float));

        // Copy input data to device
        cudaMemcpy(d_input1, input1, batch_size * num_features * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input2, input2, batch_size * num_features * sizeof(half), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

        pairwise_hamming_distance_kernel<<<numBlocks, threadsPerBlock>>>(
            d_input1, d_input2, d_output, batch_size, num_features
        );

        // Copy result back to host
        cudaMemcpy(output, d_output, batch_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input1);
        cudaFree(d_input2);
        cudaFree(d_output);
    }
}
