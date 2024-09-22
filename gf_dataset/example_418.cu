
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for scatter operation
__global__ void scatter_kernel(const float* input, const int* indices, float* output, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[indices[i] * D + threadIdx.y] += input[i * D + threadIdx.y];
    }
}

// CUDA kernel for average pooling (using cuDNN)
#include <cudnn.h>

__global__ void avgpool_kernel(const float* input, float* output, int N, int D, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        for (int j = 0; j < D; j++) {
            output[i * D + j] = input[i * D + j] / kernel_size;  // Simple averaging
        }
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

    // Extract indices tensor
    const int* indices = va_arg(args, const int*);

    // Extract dim and kernel_size
    int dim = va_arg(args, int);
    int kernel_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // --- CUDA Setup ---

    int N = input_tensor_dim0;
    int D = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    int *d_indices;
    cudaMalloc(&d_input, N * D * sizeof(float));
    cudaMalloc(&d_output, N * D * sizeof(float));
    cudaMalloc(&d_indices, N * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices, N * sizeof(int), cudaMemcpyHostToDevice);

    // --- Scatter Operation ---

    // Launch scatter kernel
    int threadsPerBlock = 128;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    scatter_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_indices, d_output, N, D);
    cudaDeviceSynchronize();  // Ensure scatter is complete

    // --- Average Pooling (using cuDNN) ---

    // Initialize cuDNN
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // Create cuDNN tensors
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);

    // Set tensor descriptions
    cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT, 1, N, D);
    cudnnSetTensorNdDescriptor(outputDesc, CUDNN_DATA_FLOAT, 1, N, D);

    // Create cuDNN pooling descriptor
    cudnnPoolingDescriptor_t poolDesc;
    cudnnCreatePoolingDescriptor(&poolDesc);
    cudnnSetPoolingNdDescriptor(poolDesc, CUDNN_POOLING_AVERAGE_CROSS_CHANNEL, CUDNN_PROPAGATE_NAN, kernel_size, 1, D);

    // Perform average pooling
    cudnnPoolingForward(handle, poolDesc, inputDesc, d_output, outputDesc, d_output);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(output, d_output, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    // --- Cleanup ---

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);

    // Destroy cuDNN resources
    cudnnDestroy(handle);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyPoolingDescriptor(poolDesc);
}

}  // extern "C"
