
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for complex max operation
__global__ void complex_max_kernel(const cufftComplex* input, float* output, int batch_size, int num_channels, int length) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch_size && channel_idx < num_channels) {
        float max_value = 0.0f;
        for (int i = 0; i < length; i++) {
            float real = input[batch_idx * num_channels * length + channel_idx * length + i].x;
            float imag = input[batch_idx * num_channels * length + channel_idx * length + i].y;
            max_value = fmaxf(max_value, fmaxf(real, imag));
        }
        output[batch_idx * num_channels * length + channel_idx * length] = max_value;
    }
}

// CUDA kernel for iRFFT
__global__ void irfft_kernel(const float* input, float* output, int batch_size, int num_channels, int length) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = batch_idx * num_channels * length + channel_idx * length;

    if (batch_idx < batch_size && channel_idx < num_channels) {
        output[idx] = input[idx];
    }
}

extern "C" {

void complex_max_irfft_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const cufftComplex* input = va_arg(args, const cufftComplex*);
    int batch_size = va_arg(args, int);
    int num_channels = va_arg(args, int);
    int length = va_arg(args, int);

    // Extract signal length
    int signal_length = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    cufftComplex *d_input;
    float *d_max_values, *d_output;
    cudaMalloc(&d_input, batch_size * num_channels * length * sizeof(cufftComplex));
    cudaMalloc(&d_max_values, batch_size * num_channels * length * sizeof(float));
    cudaMalloc(&d_output, batch_size * num_channels * signal_length * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * num_channels * length * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // Perform complex max operation
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_channels + threadsPerBlock.y - 1) / threadsPerBlock.y);
    complex_max_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_max_values, batch_size, num_channels, length);

    // Create cuFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, signal_length, CUFFT_R2C, batch_size * num_channels);

    // Perform iRFFT
    cufftExecR2C(plan, d_max_values, d_output);
    cufftDestroy(plan);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * num_channels * signal_length * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_max_values);
    cudaFree(d_output);
}

}  // extern "C"
