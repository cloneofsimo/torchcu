
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for in-place absolute value calculation
__global__ void abs_inplace_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fabsf(data[idx]);
    }
}

extern "C" {

void abs_inplace_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    float* input_tensor = va_arg(args, float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int size = batch_size * input_dim;

    // Allocate device memory
    float *d_input;
    cudaMalloc(&d_input, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    abs_inplace_kernel<<<numBlocks, threadsPerBlock>>>(d_input, size);

    // Copy result back to host (in-place)
    cudaMemcpy(input_tensor, d_input, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
}

}  // extern "C"
