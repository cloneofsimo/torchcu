
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for sorting a tensor along the first dimension
__global__ void sort_tensor_kernel(const float* input_tensor, float* output_tensor, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num_rows && col < num_cols) {
        // Calculate the index of the element in the input tensor
        int input_index = row * num_cols + col;

        // Calculate the index of the element in the output tensor
        int output_index = col * num_rows + row;

        output_tensor[output_index] = input_tensor[input_index];
    }
}

extern "C" {

void sort_tensor_by_first_dim(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output_tensor = va_arg(args, float*);

    va_end(args);

    int num_rows = input_tensor_dim0;
    int num_cols = input_tensor_dim1;

    // Allocate device memory
    float* d_input_tensor;
    float* d_output_tensor;
    cudaMalloc(&d_input_tensor, num_rows * num_cols * sizeof(float));
    cudaMalloc(&d_output_tensor, num_rows * num_cols * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input_tensor, input_tensor, num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((num_rows + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                    (num_cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sort_tensor_kernel<<<numBlocks, threadsPerBlock>>>(d_input_tensor, d_output_tensor, num_rows, num_cols);

    // Copy result back to host
    cudaMemcpy(output_tensor, d_output_tensor, num_rows * num_cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input_tensor);
    cudaFree(d_output_tensor);
}

}  // extern "C"
