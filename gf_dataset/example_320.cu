
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for hard shrink using bfloat16
__global__ void hardshrink_kernel_bf16(const float* input_tensor, float lambd, float* output, 
                                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __nv_bfloat16 input_bf16 = float_to_bfloat16(input_tensor[i]);
        __nv_bfloat16 lambd_bf16 = float_to_bfloat16(lambd);
        __nv_bfloat16 result = __hmul(input_bf16, input_bf16);
        if (bfloat16_to_float(result) > bfloat16_to_float(__hmul(lambd_bf16, lambd_bf16))) {
            output[i] = input_tensor[i];
        } else {
            output[i] = 0.0f;
        }
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

    // Extract lambda
    float lambd = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int n = input_tensor_dim0 * input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    hardshrink_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, lambd, d_output, n
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
