
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for pairwise Hamming distance calculation
__global__ void hamming_distance_int8_kernel(const int8_t* input1, const int8_t* input2, int32_t* output, int batch_size, int feature_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int32_t distance = 0;
        for (int i = 0; i < feature_size; ++i) {
            distance += __popc_ll(input1[idx * feature_size + i] ^ input2[idx * feature_size + i]);
        }
        output[idx] = distance;
    }
}

extern "C" {
    void hamming_distance_int8(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract input tensors
        const int8_t* input1 = va_arg(args, const int8_t*);
        int input1_dim0 = va_arg(args, int);
        int input1_dim1 = va_arg(args, int);

        const int8_t* input2 = va_arg(args, const int8_t*);
        int input2_dim0 = va_arg(args, int);
        int input2_dim1 = va_arg(args, int);

        // Extract output tensor
        int32_t* output = va_arg(args, int32_t*);

        va_end(args);

        int batch_size = input1_dim0;
        int feature_size = input1_dim1;

        // Allocate device memory
        int8_t *d_input1, *d_input2;
        int32_t *d_output;
        cudaMalloc(&d_input1, batch_size * feature_size * sizeof(int8_t));
        cudaMalloc(&d_input2, batch_size * feature_size * sizeof(int8_t));
        cudaMalloc(&d_output, batch_size * sizeof(int32_t));

        // Copy input data to device
        cudaMemcpy(d_input1, input1, batch_size * feature_size * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input2, input2, batch_size * feature_size * sizeof(int8_t), cudaMemcpyHostToDevice);

        // Launch kernel
        int threadsPerBlock = 256;
        int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

        hamming_distance_int8_kernel<<<numBlocks, threadsPerBlock>>>(d_input1, d_input2, d_output, batch_size, feature_size);

        // Copy result back to host
        cudaMemcpy(output, d_output, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input1);
        cudaFree(d_input2);
        cudaFree(d_output);
    }
}
