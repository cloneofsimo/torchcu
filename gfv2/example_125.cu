
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for optimized ReLU with scaling
__global__ void optimized_relu_kernel(const float* input, float* output, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = fmaxf(input[idx], 0.0f) * alpha;  // ReLU and scaling in one step
    }
}

extern "C" {

void hyperparameter_optimized_relu(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);

    // Extract alpha
    float alpha = va_arg(args, float);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int size = input_dim0 * input_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    optimized_relu_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, alpha, size);

    // Copy result back to host
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
