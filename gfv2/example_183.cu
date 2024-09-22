
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void outer_product_int8_kernel(const float* input_tensor, int8_t* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        output[i * n + j] = (int8_t)(input_tensor[i] * input_tensor[j]);
    }
}

extern "C" {

void outer_product_int8(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    int8_t* output = va_arg(args, int8_t*);

    va_end(args);

    int n = input_tensor_dim0;

    // Allocate device memory
    float* d_input;
    int8_t* d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * n * sizeof(int8_t));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    outer_product_int8_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, n);

    // Copy result back to host
    cudaMemcpy(output, d_output, n * n * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
