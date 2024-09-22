
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for element-wise minimum
__global__ void elementwise_min_kernel(const float* input1, const float* input2, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fminf(input1[idx], input2[idx]);
    }
}

extern "C" {

void elementwise_min_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input1 = va_arg(args, const float*);
    int input1_dim0 = va_arg(args, int);
    int input1_dim1 = va_arg(args, int);

    const float* input2 = va_arg(args, const float*);
    int input2_dim0 = va_arg(args, int);
    int input2_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Calculate the total size of the input tensors
    int size = input1_dim0 * input1_dim1;

    // Allocate device memory
    float *d_input1, *d_input2, *d_output;
    cudaMalloc(&d_input1, size * sizeof(float));
    cudaMalloc(&d_input2, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input1, input1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_min_kernel<<<numBlocks, threadsPerBlock>>>(d_input1, d_input2, d_output, size);

    // Copy result back to host
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}

}  // extern "C"
