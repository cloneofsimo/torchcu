
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

extern "C" {

void true_divide_and_relu(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract divisor
    float divisor = va_arg(args, float);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int total_elements = batch_size * input_dim;

    // Allocate device memory
    float *d_input;
    cudaMalloc(&d_input, total_elements * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, total_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 threadsPerBlock(256);
    dim3 numBlocks((total_elements + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Kernel for true division and ReLU
    __global__ void true_divide_relu_kernel(float *input, float divisor, int total_elements) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_elements) {
            input[idx] = fmaxf(input[idx] / divisor, 0.0f);
        }
    }

    // Launch kernel
    true_divide_relu_kernel<<<numBlocks, threadsPerBlock>>>(d_input, divisor, total_elements);

    // Copy result back to host
    cudaMemcpy(input_tensor, d_input, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
}
}
