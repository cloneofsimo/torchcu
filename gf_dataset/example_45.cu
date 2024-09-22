
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for median filter
__global__ void median_filter_kernel(const float* input, float* output, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        int start = max(0, i - K / 2);
        int end = min(N, i + K / 2 + 1);

        // Create a temporary array for the window
        float window[K];
        for (int j = start, k = 0; j < end; ++j, ++k) {
            window[k] = input[j];
        }

        // Sort the window
        for (int j = 0; j < K; ++j) {
            for (int k = j + 1; k < K; ++k) {
                if (window[j] > window[k]) {
                    float tmp = window[j];
                    window[j] = window[k];
                    window[k] = tmp;
                }
            }
        }

        // Assign the median value to the output
        output[i] = window[K / 2];
    }
}

// CUDA kernel for 1D convolution
__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, 
                              int N, int K, int padding) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float sum = 0.0f;
        for (int j = -padding; j <= padding; ++j) {
            int idx = i + j;
            if (idx >= 0 && idx < N) {
                sum += input[idx] * kernel[j + padding];
            }
        }
        output[i] = sum;
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

        // Extract kernel size
        int kernel_size = va_arg(args, int);

        // Extract output tensor (assuming it's preallocated)
        float* output = va_arg(args, float*);

        va_end(args);

        int batch_size = input_tensor_dim0;
        int input_dim = input_tensor_dim1;
        int padding = kernel_size / 2;

        // Allocate device memory
        float* d_input, *d_output, *d_kernel;
        cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
        cudaMalloc(&d_output, batch_size * input_dim * sizeof(float));
        cudaMalloc(&d_kernel, kernel_size * sizeof(float));

        // Copy input and kernel data to device
        cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, &kernel_size, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch median filter kernel
        dim3 threadsPerBlock(128);
        dim3 numBlocks((input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x);
        median_filter_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, input_dim, kernel_size);

        // Launch convolution kernel
        conv1d_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_kernel, d_output, input_dim, kernel_size, padding);

        // Copy result back to host
        cudaMemcpy(output, d_output, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);
    }
}
