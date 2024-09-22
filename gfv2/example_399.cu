
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for pairwise Hamming distance and addition
__global__ void pairwise_hamming_distance_add_kernel(const float* input_tensor, const float* weights, float* output, 
                                        int batch_size, int feature_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < feature_dim) {
        int hamming_distance = 0;
        for (int i = 0; i < 8; ++i) { // Assuming 8-bit integer representation
            int input_val = ((int)input_tensor[row * feature_dim + col] >> i) & 1;
            int weight_val = ((int)weights[col] >> i) & 1;
            if (input_val != weight_val) {
                hamming_distance++;
            }
        }
        output[row * feature_dim + col] = input_tensor[row * feature_dim + col] + hamming_distance;
    }
}

extern "C" {

void pairwise_hamming_distance_add(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weights tensor
    const float* weights = va_arg(args, const float*);
    int weights_dim0 = va_arg(args, int); 

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int feature_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_weights, *d_output;
    cudaMalloc(&d_input, batch_size * feature_dim * sizeof(float));
    cudaMalloc(&d_weights, feature_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * feature_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, feature_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((feature_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    pairwise_hamming_distance_add_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weights, d_output, batch_size, feature_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * feature_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
}

}  // extern "C"
