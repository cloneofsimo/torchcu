
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdarg.h> 

#define PI 3.14159265358979323846

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for DFT
__global__ void dft_kernel_bf16(const float *input, float *output, int batch_size, int signal_length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < batch_size && j < signal_length) {
        float sum_real = 0.0f;
        float sum_imag = 0.0f;
        for (int k = 0; k < signal_length; ++k) {
            float angle = -2.0f * PI * (float)j * (float)k / (float)signal_length;
            __nv_bfloat16 real_part = float_to_bfloat16(cosf(angle));
            __nv_bfloat16 imag_part = float_to_bfloat16(sinf(angle));
            __nv_bfloat16 input_bf16 = float_to_bfloat16(input[i * signal_length + k]);
            
            // Compute complex multiplication in bfloat16
            __nv_bfloat16 real_result = __hmul(real_part, input_bf16) - __hmul(imag_part, input_bf16);
            __nv_bfloat16 imag_result = __hmul(real_part, input_bf16) + __hmul(imag_part, input_bf16);
            
            sum_real += bfloat16_to_float(real_result);
            sum_imag += bfloat16_to_float(imag_result);
        }
        output[i * signal_length + j] = sum_real; // Store real part
        output[i * signal_length + j + signal_length * batch_size] = sum_imag; // Store imaginary part
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int signal_length = va_arg(args, int);

    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * signal_length * sizeof(float));
    cudaMalloc(&d_output, batch_size * signal_length * 2 * sizeof(float)); // 2 for real & imag

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * signal_length * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((signal_length + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    dft_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, signal_length
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * signal_length * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}
