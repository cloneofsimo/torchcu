
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

// CUDA kernel for softmax calculation on the diagonal
__global__ void softmax_diag_kernel_bf16(const float* input_tensor, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            __nv_bfloat16 val = float_to_bfloat16(input_tensor[i * n + j]);
            sum += expf(bfloat16_to_float(val));
        }
        output[i] = expf(bfloat16_to_float(float_to_bfloat16(input_tensor[i * n + i]))) / sum;
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

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int n = input_tensor_dim0;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel for diagonal softmax
    dim3 threadsPerBlock(256);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    softmax_diag_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, n
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Update the diagonal of the input tensor on the device
    for (int i = 0; i < n; ++i) {
        d_input[i * n + i] = d_output[i];
    }

    // Copy the modified input tensor back to the host
    cudaMemcpy(input_tensor, d_input, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
