
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

// CUDA kernel for cumprod using bfloat16 (using a simple implementation)
__global__ void cumprod_kernel_bf16(const float* input_tensor, float* output, 
                                        int batch_size, int seq_length, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_length) {
        int row = idx / seq_length;
        int col = idx % seq_length;

        __nv_bfloat16 product = float_to_bfloat16(input_tensor[row * seq_length + col]);
        if (dim == 1) {
            for (int i = 0; i < col; ++i) {
                product = __hmul(product, float_to_bfloat16(input_tensor[row * seq_length + i]));
            }
        } else {
            for (int i = 0; i < row; ++i) {
                product = __hmul(product, float_to_bfloat16(input_tensor[i * seq_length + col]));
            }
        }
        output[idx] = bfloat16_to_float(product);
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

    // Extract dim
    int dim = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int seq_length = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * seq_length * sizeof(float));
    cudaMalloc(&d_output, batch_size * seq_length * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * seq_length * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size * seq_length + threadsPerBlock.x - 1) / threadsPerBlock.x);

    cumprod_kernel_bf16<<<numBlocks, threadsPerBlock>>>(d_input, d_output, batch_size, seq_length, dim);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * seq_length * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
