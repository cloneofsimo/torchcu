
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

__global__ void frobenius_norm_kernel(const float* input, float* output, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < m * n) {
        int row = idx / n;
        int col = idx % n;

        __nv_bfloat16 val = float_to_bfloat16(input[row * n + col]);
        output[0] += bfloat16_to_float(__hmul(val, val));
    }
}

extern "C" {

void frobenius_norm_with_meshgrid(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int m = input_dim0;
    int n = input_dim1;

    // Allocate device memory
    float* d_input, *d_output;
    cudaMalloc(&d_input, m * n * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((m * n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    frobenius_norm_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, m, n);

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Calculate the square root for final result
    output[0] = sqrtf(output[0]);
}

} // extern "C"
