
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void addcmul_kernel(const float* input_tensor, const float* tensor1, const float* tensor2, float value, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = input_tensor[idx] + value * tensor1[idx] * tensor2[idx];
    }
}

extern "C" {
void addcmul_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    const float* tensor1 = va_arg(args, const float*);
    int tensor1_dim0 = va_arg(args, int);
    int tensor1_dim1 = va_arg(args, int);

    const float* tensor2 = va_arg(args, const float*);
    int tensor2_dim0 = va_arg(args, int);
    int tensor2_dim1 = va_arg(args, int);

    float value = va_arg(args, double);

    float* output = va_arg(args, float*);

    va_end(args);

    int size = input_tensor_dim0 * input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_tensor1, *d_tensor2, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_tensor1, size * sizeof(float));
    cudaMalloc(&d_tensor2, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tensor1, tensor1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tensor2, tensor2, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    addcmul_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_tensor1, d_tensor2, value, d_output, size);

    // Copy result back to host
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_tensor1);
    cudaFree(d_tensor2);
    cudaFree(d_output);
}
}
