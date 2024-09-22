
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <complex>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for DFT using bfloat16
__global__ void dft_bf16_kernel(const float* input, std::complex<float>* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        __nv_bfloat16 sum_real = 0.0f;
        __nv_bfloat16 sum_imag = 0.0f;
        for (int k = 0; k < N; ++k) {
            float angle = 2 * M_PI * i * k / N;
            __nv_bfloat16 real_part = float_to_bfloat16(cos(angle));
            __nv_bfloat16 imag_part = float_to_bfloat16(sin(angle));
            __nv_bfloat16 input_bf16 = float_to_bfloat16(input[k]);
            sum_real += __hmul(input_bf16, real_part);
            sum_imag += __hmul(input_bf16, imag_part);
        }
        output[i] = std::complex<float>(bfloat16_to_float(sum_real), bfloat16_to_float(sum_imag));
    }
}

extern "C" {

void dft_bf16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);

    float* output_tensor = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input;
    std::complex<float> *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * sizeof(float));
    cudaMalloc(&d_output, input_tensor_dim0 * sizeof(std::complex<float>));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int num_blocks = (input_tensor_dim0 + 255) / 256;
    dft_bf16_kernel<<<num_blocks, 256>>>(d_input, d_output, input_tensor_dim0);

    // Copy result back to host
    cudaMemcpy(output_tensor, d_output, input_tensor_dim0 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

} // extern "C"
