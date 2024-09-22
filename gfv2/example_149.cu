
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

extern "C" {

void int8_add_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const int* input_tensor1 = va_arg(args, const int*);
    int input_tensor1_dim0 = va_arg(args, int);
    int input_tensor1_dim1 = va_arg(args, int);

    const int* input_tensor2 = va_arg(args, const int*);
    int input_tensor2_dim0 = va_arg(args, int);
    int input_tensor2_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    int* output = va_arg(args, int*);

    va_end(args);

    // Allocate device memory
    int *d_input1, *d_input2, *d_output;
    cudaMalloc(&d_input1, input_tensor1_dim0 * input_tensor1_dim1 * sizeof(int));
    cudaMalloc(&d_input2, input_tensor2_dim0 * input_tensor2_dim1 * sizeof(int));
    cudaMalloc(&d_output, input_tensor1_dim0 * input_tensor1_dim1 * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input1, input_tensor1, input_tensor1_dim0 * input_tensor1_dim1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input_tensor2, input_tensor2_dim0 * input_tensor2_dim1 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((input_tensor1_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_tensor1_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    __global__ void add_kernel(const int* input1, const int* input2, int* output, int m, int n) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < n) {
            output[row * n + col] = input1[row * n + col] + input2[row * n + col];
        }
    }

    add_kernel<<<numBlocks, threadsPerBlock>>>(d_input1, d_input2, d_output, input_tensor1_dim0, input_tensor1_dim1);

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor1_dim0 * input_tensor1_dim1 * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}

}  // extern "C"
