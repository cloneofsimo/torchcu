
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for batch matrix multiplication with addition
__global__ void baddbmm_kernel(const float* input_tensor, const float* batch1, 
                               const float* batch2, float* output,
                               int batch_size, int m, int n, int k) {

    int b = blockIdx.z; // Batch index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && row < m && col < n) {
        float sum = input_tensor[b * m * n + row * n + col]; // Load from input tensor
        for (int i = 0; i < k; ++i) {
            sum += batch1[b * m * k + row * k + i] * batch2[b * k * n + i * n + col];
        }
        output[b * m * n + row * n + col] = sum;
    }
}

extern "C" {

void baddbmm_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract batch1 tensor
    const float* batch1 = va_arg(args, const float*);
    int batch1_dim0 = va_arg(args, int);
    int batch1_dim1 = va_arg(args, int);
    int batch1_dim2 = va_arg(args, int);

    // Extract batch2 tensor
    const float* batch2 = va_arg(args, const float*);
    int batch2_dim0 = va_arg(args, int);
    int batch2_dim1 = va_arg(args, int);
    int batch2_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = batch1_dim0; // Assuming batch1 and batch2 have the same batch size
    int m = input_tensor_dim1;
    int n = input_tensor_dim2;
    int k = batch1_dim2; // Assuming batch1 and batch2 have the same inner dimension

    // Allocate device memory
    float *d_input, *d_batch1, *d_batch2, *d_output;
    cudaMalloc(&d_input, batch_size * m * n * sizeof(float));
    cudaMalloc(&d_batch1, batch_size * m * k * sizeof(float));
    cudaMalloc(&d_batch2, batch_size * k * n * sizeof(float));
    cudaMalloc(&d_output, batch_size * m * n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_batch1, batch1, batch_size * m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_batch2, batch2, batch_size * k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   batch_size);

    baddbmm_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_batch1, d_batch2, d_output,
                                                    batch_size, m, n, k);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_batch1);
    cudaFree(d_batch2);
    cudaFree(d_output);
}

}  // extern "C"
