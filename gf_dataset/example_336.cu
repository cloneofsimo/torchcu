
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

// Helper function to convert float to __half
__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half_rn(f); 
}

// Helper function to convert __half to float
__device__ __forceinline__ float half_to_float(__half h) {
    return __half2float(h);
}

// CUDA kernel for calculating standard deviation
__global__ void std_kernel_fp16(const float* input, float* output, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j < m; j++) {
            __half val = float_to_half(input[i * m + j]);
            sum += half_to_float(val);
        }

        float mean = sum / m;
        float sq_sum = 0.0f;

        for (int j = 0; j < m; j++) {
            __half val = float_to_half(input[i * m + j]);
            float diff = half_to_float(val) - mean;
            sq_sum += diff * diff;
        }

        output[i] = sqrtf(sq_sum / (m - 1));
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

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    std_kernel_fp16<<<(batch_size + 255) / 256, 256>>>(d_input, d_output, batch_size, input_dim);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

} // extern "C"
