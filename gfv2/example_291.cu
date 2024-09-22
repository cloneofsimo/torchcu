
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {

void my_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * sizeof(float));
    cudaMalloc(&d_output, input_tensor_dim0 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((input_tensor_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x);

    my_function_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, input_tensor_dim0);

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor_dim0 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void my_function_kernel(const float* input_tensor, float* output, int dim0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim0) {
        output[i] = (input_tensor[i] * 2.0f) + 3.0f;
    }
}

}
