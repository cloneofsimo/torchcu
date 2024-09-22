
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for max filtering and adaptive max pooling
__global__ void max_filter_adaptive_pool_kernel_fp16(const half* input, half* output, 
                                                         int batch_size, int input_size, int kernel_size, int output_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Max filtering with kernel size
        int start_idx = max(0, batch_idx - (kernel_size / 2));
        int end_idx = min(batch_idx + (kernel_size / 2) + 1, input_size);
        half max_value = input[start_idx];
        for (int i = start_idx + 1; i < end_idx; ++i) {
            max_value = max(max_value, input[i]);
        }
        
        // Adaptive max pooling
        int output_idx = (int) ( ((float)batch_idx / input_size) * output_size );
        atomicMax(&output[output_idx], max_value);
    }
}

extern "C" {

void max_filter_adaptive_pool_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract kernel size
    int kernel_size = va_arg(args, int);

    // Extract output size
    int output_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    half *d_input, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(half));
    cudaMalloc(&d_output, input_tensor_dim0 * output_size * sizeof(half));

    // Copy input data to device (converted to half)
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32);
    dim3 numBlocks((input_tensor_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x);

    max_filter_adaptive_pool_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, input_tensor_dim0, input_tensor_dim1, kernel_size, output_size
    );

    // Copy result back to host (converted to float)
    cudaMemcpy(output, d_output, input_tensor_dim0 * output_size * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
