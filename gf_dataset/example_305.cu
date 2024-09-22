
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

#include <iostream>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// CUDA kernel for Hadamard product and any using fp16
__global__ void hadamard_any_kernel_fp16(const float* input1, const float* input2, 
                                        const int* shape, bool* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        half a = float_to_half(input1[idx]);
        half b = float_to_half(input2[idx]);
        half result = __hmul(a, b);
        *output = *output || (result != 0.0f);
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input1 = va_arg(args, const float*);
    int input1_dim0 = va_arg(args, int);
    int input1_dim1 = va_arg(args, int);

    const float* input2 = va_arg(args, const float*);
    int input2_dim0 = va_arg(args, int);
    int input2_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    bool* output = va_arg(args, bool*);

    va_end(args);

    int batch_size = input1_dim0;
    int input_dim = input1_dim1;

    // Allocate device memory
    float *d_input1, *d_input2;
    bool *d_output;
    cudaMalloc(&d_input1, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_input2, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, sizeof(bool));

    // Copy input data to device
    cudaMemcpy(d_input1, input1, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize output on device to false
    cudaMemset(d_output, 0, sizeof(bool));

    // Launch kernel
    int num_elements = batch_size * input_dim;
    hadamard_any_kernel_fp16<<<(num_elements + 255) / 256, 256>>>(
        d_input1, d_input2, nullptr, d_output, num_elements
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(bool), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}

}  // extern "C"
