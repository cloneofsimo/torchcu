
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for Softmin function
__global__ void softmin_kernel_bf16(const float* input_tensor, float* output, int batch_size, int dim, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < size) {
        float sum = 0.0f;
        __nv_bfloat16 val = float_to_bfloat16(input_tensor[row * size + col]);
        for (int i = 0; i < size; ++i) {
            __nv_bfloat16 temp = float_to_bfloat16(input_tensor[row * size + i]);
            temp = __hmul(temp, expf(logspace_values[col])); // Apply logspace values
            sum += bfloat16_to_float(__hextf(temp));
        }
        output[row * size + col] = bfloat16_to_float(__hmul(val, expf(-sum)));  // Softmin calculation
    }
}

// Global variable for logspace values
__constant__ __nv_bfloat16 logspace_values[10]; // Adjust the size based on your input tensor's size(dim)

extern "C" {

void softmin_logspace_bf16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * dim * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize logspace values (make sure this matches your input tensor's size(dim))
    __nv_bfloat16 temp_logspace_values[10];
    for (int i = 0; i < dim; ++i) {
        temp_logspace_values[i] = float_to_bfloat16(expf(logf(i + 1) / 10));
    }
    cudaMemcpyToSymbol(logspace_values, temp_logspace_values, dim * sizeof(__nv_bfloat16), 0, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    softmin_kernel_bf16<<<numBlocks, threadsPerBlock>>>(d_input, d_output, batch_size, dim, dim);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
