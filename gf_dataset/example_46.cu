
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cufft.h>  // Include cuFFT library
#include <stdarg.h>

extern "C" {

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f); 
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

__global__ void fft_conv1d_kernel(const float* input, const float* weight, float* output, 
                                    int batch_size, int input_length, int kernel_length, 
                                    int output_length) {
    int b = blockIdx.x;
    int i = threadIdx.x;

    if (b < batch_size && i < output_length) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_length; ++k) {
            if (i - k >= 0) {
                sum += half_to_float(float_to_half(input[b * input_length + (i - k)]) * 
                                    float_to_half(weight[k]));
            }
        }
        output[b * output_length + i] = sum;
    }
}

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);
    int input_dim2 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    // Calculate output length for valid convolution
    int output_length = input_dim1 - weight_dim2 + 1;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, input_dim0 * input_dim1 * input_dim2 * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * weight_dim2 * sizeof(float));
    cudaMalloc(&d_output, input_dim0 * output_length * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, input_dim0 * input_dim1 * input_dim2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * weight_dim2 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(output_length, 1);
    dim3 numBlocks(input_dim0, 1);

    fft_conv1d_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_output, 
                                                    input_dim0, input_dim1, weight_dim2, output_length);

    // Copy result back to host
    cudaMemcpy(output, d_output, input_dim0 * output_length * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}
}  // extern "C"
