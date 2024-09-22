
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for pairwise Hamming distance calculation
__global__ void hamming_distance_kernel(const float* input_tensor, const float* weight, float* output,
                                        int batch_size, int num_weights, int feature_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < num_weights) {
        float distance = 0.0f;
        for (int i = 0; i < feature_dim; ++i) {
            distance += abs(input_tensor[row * feature_dim + i] - weight[col * feature_dim + i]);
        }
        output[row * num_weights + col] = distance;
    }
}

extern "C" {

void hamming_distance_layer(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int feature_dim = input_tensor_dim1;
    int num_weights = weight_dim0;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, batch_size * feature_dim * sizeof(float));
    cudaMalloc(&d_weight, num_weights * feature_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * num_weights * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, num_weights * feature_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((num_weights + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    hamming_distance_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, batch_size, num_weights, feature_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * num_weights * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

} // extern "C"
