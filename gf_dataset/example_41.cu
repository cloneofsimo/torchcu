
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

__global__ void addcmul_kernel(const float* input1, const float* input2, const float* weight, float value, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = input1[i] + value * input2[i] * weight[i];
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input1 = va_arg(args, const float*);
    int input1_dim0 = va_arg(args, int);
    int input1_dim1 = va_arg(args, int);

    const float* input2 = va_arg(args, const float*);
    int input2_dim0 = va_arg(args, int);
    int input2_dim1 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    float value = va_arg(args, double); // Extract the float value

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int size = input1_dim0 * input1_dim1;

    // Allocate device memory
    float *d_input1, *d_input2, *d_weight, *d_output;
    cudaMalloc(&d_input1, size * sizeof(float));
    cudaMalloc(&d_input2, size * sizeof(float));
    cudaMalloc(&d_weight, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input1, input1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    addcmul_kernel<<<numBlocks, threadsPerBlock>>>(d_input1, d_input2, d_weight, value, d_output, size);

    // Copy result back to host
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_weight);
    cudaFree(d_output);
}

} // extern "C"
