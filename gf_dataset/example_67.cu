
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

// CUDA kernel for hardtanh activation
__global__ void hardtanh_kernel(const float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = float_to_half(fmaxf(fminf(val, 1.0f), -1.0f));
    }
}

// CUDA kernel for finding unique elements and storing them in fp16
__global__ void unique_fp16_kernel(const half* input, half* output, int* num_unique, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Check if this element has been encountered before
        bool unique = true;
        for (int j = 0; j < idx; ++j) {
            if (input[idx] == input[j]) {
                unique = false;
                break;
            }
        }
        // Store unique element in output array
        if (unique) {
            output[atomicAdd(num_unique, 1)] = input[idx]; 
        }
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int input_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    half* output = va_arg(args, half*);

    va_end(args);

    // Allocate device memory
    half *d_input, *d_output;
    int *d_num_unique;
    cudaMalloc(&d_input, input_size * sizeof(half));
    cudaMalloc(&d_output, input_size * sizeof(half));
    cudaMalloc(&d_num_unique, sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch hardtanh kernel
    hardtanh_kernel<<<(input_size + 255) / 256, 256>>>(d_input, d_output, input_size);

    // Initialize num_unique to 0
    cudaMemset(d_num_unique, 0, sizeof(int));

    // Launch unique_fp16 kernel
    unique_fp16_kernel<<<(input_size + 255) / 256, 256>>>(d_output, d_output, d_num_unique, input_size);

    // Copy result back to host
    int num_unique;
    cudaMemcpy(&num_unique, d_num_unique, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(output, d_output, num_unique * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_num_unique);
}

}  // extern "C"
