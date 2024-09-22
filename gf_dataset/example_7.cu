
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for cholesky decomposition and average pooling
__global__ void cholesky_pool_kernel(const float* input_tensor, float* output, 
                                      int batch_size, int input_channels, int input_length,
                                      int kernel_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int output_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (batch_idx < batch_size && channel_idx < input_channels && output_idx < (input_length - kernel_size + 1)) {
        // Cholesky decomposition
        half* L = (half*)malloc(input_length * sizeof(half));  // Allocate on device for each thread
        for (int i = 0; i < input_length; i++) {
            L[i] = float_to_half(input_tensor[batch_idx * input_channels * input_length + channel_idx * input_length + i]);
        }
        // Calculate the lower triangular Cholesky factor
        for (int i = 0; i < input_length; i++) {
            for (int j = 0; j < i; j++) {
                L[i] -= L[j] * L[j];
            }
            L[i] = sqrtf(half_to_float(L[i]));
        }
        // Average pooling
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; i++) {
            sum += half_to_float(L[output_idx + i]);
        }
        output[batch_idx * input_channels * (input_length - kernel_size + 1) + channel_idx * (input_length - kernel_size + 1) + output_idx] = sum / kernel_size;
        free(L);  // Free device memory allocated by each thread
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
    int input_tensor_dim2 = va_arg(args, int);

    // Extract kernel size
    int kernel_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_channels = input_tensor_dim1;
    int input_length = input_tensor_dim2;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_channels * input_length * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_channels * (input_length - kernel_size + 1) * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_channels * input_length * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (input_length - kernel_size + 1 + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cholesky_pool_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, input_channels, input_length, kernel_size
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_channels * (input_length - kernel_size + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
