
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// This implementation uses a custom CUDA kernel for CELU and a separate kernel for average pooling

// CUDA kernel for CELU activation (using FP16)
__global__ void celu_kernel(const half* input, half* output, int batch_size, int channels, int height, int width, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels * height * width) {
        int b = idx / (channels * height * width);
        int c = (idx % (channels * height * width)) / (height * width);
        int h = (idx % (height * width)) / width;
        int w = idx % width;

        float val = __int_as_float(input[idx]);  // Convert from FP16 to FP32
        output[idx] = __float2half_rn(val <= 0.0f ? alpha * (exp(val / alpha) - 1.0f) : val);  // Apply CELU
    }
}

// CUDA kernel for average pooling (using FP16)
__global__ void avg_pool2d_kernel(const half* input, half* output, int batch_size, int channels, int height, int width, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels * (height / kernel_size) * (width / kernel_size)) {
        int b = idx / (channels * (height / kernel_size) * (width / kernel_size));
        int c = (idx % (channels * (height / kernel_size) * (width / kernel_size))) / ((height / kernel_size) * (width / kernel_size));
        int h = (idx % ((height / kernel_size) * (width / kernel_size))) / (width / kernel_size);
        int w = idx % (width / kernel_size);

        float sum = 0.0f;
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                sum += __int_as_float(input[b * channels * height * width + c * height * width + (h * kernel_size + i) * width + w * kernel_size + j]);
            }
        }
        output[idx] = __float2half_rn(sum / (kernel_size * kernel_size));
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
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;
    int kernel_size = 2; // for average pooling

    // Allocate device memory
    half *d_input, *d_output_celu, *d_output_pool;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(half));
    cudaMalloc(&d_output_celu, batch_size * channels * height * width * sizeof(half));
    cudaMalloc(&d_output_pool, batch_size * channels * (height / kernel_size) * (width / kernel_size) * sizeof(half));

    // Copy input data to device (convert to FP16)
    cudaMemcpy(d_input, input_tensor, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CELU kernel
    dim3 threadsPerBlock(128);
    dim3 numBlocks((batch_size * channels * height * width + threadsPerBlock.x - 1) / threadsPerBlock.x);
    celu_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output_celu, batch_size, channels, height, width, 1.0f);

    // Launch average pooling kernel
    threadsPerBlock = dim3(128);
    numBlocks = dim3((batch_size * channels * (height / kernel_size) * (width / kernel_size) + threadsPerBlock.x - 1) / threadsPerBlock.x);
    avg_pool2d_kernel<<<numBlocks, threadsPerBlock>>>(d_output_celu, d_output_pool, batch_size, channels, height, width, kernel_size);

    // Copy result back to host (convert back to FP32)
    cudaMemcpy(output, d_output_pool, batch_size * channels * (height / kernel_size) * (width / kernel_size) * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output_celu);
    cudaFree(d_output_pool);
}

}  // extern "C"
