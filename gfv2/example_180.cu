
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

// CUDA kernel for FFT shift
__global__ void fft_shift_kernel(const __int8_t* input, float* output, int n, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        int index = i * n + j;
        int shifted_index = (i * n + (j + (n / 2)) % n) % (n * m);

        // Re-arrange elements for FFT shift
        output[shifted_index] = input[index];
    }
}

// CUDA kernel for calculating the k-th value after FFT shift
__global__ void kth_value_kernel(const float* input, float* output, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        output[i] = input[i * (k + 1)]; // Accessing the k-th value
    }
}

extern "C" {

void kth_value_fft_shift(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract k value
    int k = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // CUDA memory allocations for input, output, and temporary arrays
    __int8_t *d_input, *d_fft_input, *d_fft_shifted;
    float *d_output, *d_fft_output, *d_shifted_output;

    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(__int8_t));
    cudaMalloc(&d_fft_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(__int8_t));
    cudaMalloc(&d_fft_shifted, input_tensor_dim0 * input_tensor_dim1 * sizeof(__int8_t));
    cudaMalloc(&d_fft_output, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
    cudaMalloc(&d_shifted_output, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
    cudaMalloc(&d_output, input_tensor_dim0 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Copy input data to temporary array (to avoid modification of original input)
    cudaMemcpy(d_fft_input, d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(__int8_t), cudaMemcpyDeviceToDevice);

    // Launch FFT shift kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_tensor_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (input_tensor_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    fft_shift_kernel<<<numBlocks, threadsPerBlock>>>(d_fft_input, d_fft_shifted, input_tensor_dim1, input_tensor_dim0, k);

    // Launch k-th value kernel
    numBlocks = (input_tensor_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    kth_value_kernel<<<numBlocks, threadsPerBlock>>>(d_fft_shifted, d_output, input_tensor_dim0, k);

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor_dim0 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_fft_input);
    cudaFree(d_fft_shifted);
    cudaFree(d_fft_output);
    cudaFree(d_shifted_output);
    cudaFree(d_output);
}

}  // extern "C"
