
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to perform max pooling on a block
__device__ float max_pool_block(const float* data, int kernel_size) {
    float max_value = data[0];
    for (int i = 1; i < kernel_size; ++i) {
        max_value = fmaxf(max_value, data[i]);
    }
    return max_value;
}

// CUDA kernel for 1D max pooling followed by unfold
__global__ void pool_unfold_kernel(const float* input, float* output, int batch_size, int input_size,
                                   int kernel_size, int stride, int output_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < output_size) {
        int input_index = row * input_size + col * stride;
        output[row * output_size + col] = max_pool_block(&input[input_index], kernel_size);
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract kernel_size
    int kernel_size = va_arg(args, int);

    // Extract stride
    int stride = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_size = input_tensor_dim1;

    int output_size = (input_size - kernel_size) / stride + 1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    pool_unfold_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, batch_size, input_size,
                                                     kernel_size, stride, output_size);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

} // extern "C"
