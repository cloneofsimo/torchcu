
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for softmin calculation
__global__ void softmin_kernel(const float* input_tensor, float* output, int dim, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int index = row * n + col;
        float maxVal = input_tensor[index];
        for (int i = 0; i < n; ++i) {
            if (input_tensor[row * n + i] > maxVal) {
                maxVal = input_tensor[row * n + i];
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += expf(-input_tensor[row * n + i] + maxVal);
        }

        output[index] = expf(-input_tensor[index] + maxVal) / sum;
    }
}

extern "C" {

void softmin_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract dim
    int dim = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int m = input_tensor_dim0;
    int n = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, m * n * sizeof(float));
    cudaMalloc(&d_output, m * n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    softmin_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, dim, m, n
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
