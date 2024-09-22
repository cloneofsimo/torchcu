
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

// CUDA kernel for inplace power operation
__global__ void pow_inplace_kernel(half* input_tensor, float exponent, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input_tensor[idx] = half_to_float(powf(half_to_float(input_tensor[idx]), exponent));  // Inplace operation
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

    // Extract exponent
    float exponent = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int size = input_tensor_dim0 * input_tensor_dim1;

    // Allocate device memory
    half* d_input;
    cudaMalloc(&d_input, size * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    pow_inplace_kernel<<<numBlocks, threadsPerBlock>>>(d_input, exponent, size);

    // Copy result back to host
    cudaMemcpy(output, d_input, size * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
}

}  // extern "C"
