
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for regularized roll, int8 to FP16 conversion
__global__ void regularized_roll_int8_fp16_kernel(const int8_t* input, const int shift, const float weight, half* output, 
                                                    int batch_size, int seq_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size * seq_len) {
        int row = i / seq_len;
        int col = i % seq_len;

        int new_col = (col + shift) % seq_len; // Calculate rolled index
        int rolled_idx = row * seq_len + new_col;

        // Regularization
        float mean = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            mean += input[row * seq_len + j];
        }
        mean /= seq_len;

        // Calculate and store the result
        output[i] = __int2half_rn(input[rolled_idx] - (mean * weight));
    }
}

extern "C" {

void regularized_roll_int8_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const int8_t* input = va_arg(args, const int8_t*);
    int batch_size = va_arg(args, int);
    int seq_len = va_arg(args, int);

    int shift = va_arg(args, int);

    float weight = va_arg(args, float);

    half* output = va_arg(args, half*);

    va_end(args);

    // Allocate device memory
    int8_t* d_input;
    half* d_output;
    cudaMalloc(&d_input, batch_size * seq_len * sizeof(int8_t));
    cudaMalloc(&d_output, batch_size * seq_len * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * seq_len * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256); 
    dim3 numBlocks((batch_size * seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x);

    regularized_roll_int8_fp16_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, shift, weight, d_output, batch_size, seq_len
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * seq_len * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

} // extern "C"
