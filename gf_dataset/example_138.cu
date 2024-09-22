
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to complex
__device__ __forceinline__ cufftComplex float_to_complex(float f) {
    cufftComplex c;
    c.x = f;
    c.y = 0.0f;
    return c;
}

// CUDA kernel for convolution
__global__ void conv_kernel(const cufftComplex* input, const cufftComplex* kernel, cufftComplex* output,
                            int batch_size, int input_size, int kernel_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && i < input_size) {
        cufftComplex sum = make_cufftComplex(0.0f, 0.0f);
        for (int k = 0; k < kernel_size; ++k) {
            int j = i - k;
            if (j >= 0 && j < input_size) {
                sum.x += input[b * input_size + j].x * kernel[k].x - input[b * input_size + j].y * kernel[k].y;
                sum.y += input[b * input_size + j].x * kernel[k].y + input[b * input_size + j].y * kernel[k].x;
            }
        }
        output[b * input_size + i] = sum;
    }
}

// CUDA kernel for iFFT
__global__ void ifft_kernel(const cufftComplex* input, float* output, int batch_size, int input_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && i < input_size) {
        output[b * input_size + i] = input[b * input_size + i].x;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor_real = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract kernel tensor
    const float* kernel_real = va_arg(args, const float*);
    int kernel_dim0 = va_arg(args, int);
    int kernel_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_size = input_tensor_dim1;
    int kernel_size = kernel_dim1;

    // Allocate device memory
    cufftComplex* d_input, *d_kernel, *d_output;
    float* d_output_real;
    cudaMalloc(&d_input, batch_size * input_size * sizeof(cufftComplex));
    cudaMalloc(&d_kernel, kernel_size * sizeof(cufftComplex));
    cudaMalloc(&d_output, batch_size * input_size * sizeof(cufftComplex));
    cudaMalloc(&d_output_real, batch_size * input_size * sizeof(float));

    // Copy input and kernel data to device
    for (int i = 0; i < batch_size * input_size; ++i) {
        d_input[i] = float_to_complex(input_tensor_real[i]);
    }
    for (int i = 0; i < kernel_size; ++i) {
        d_kernel[i] = float_to_complex(kernel_real[i]);
    }

    // Launch convolution kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conv_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_kernel, d_output, batch_size, input_size, kernel_size);

    // Perform iFFT using cuFFT
    cufftHandle plan;
    cufftPlan1d(&plan, input_size, CUFFT_C2R, batch_size);
    cufftExecC2R(plan, d_output, d_output_real);
    cufftDestroy(plan);

    // Copy result back to host
    cudaMemcpy(output, d_output_real, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_output_real);
}

}  // extern "C"
