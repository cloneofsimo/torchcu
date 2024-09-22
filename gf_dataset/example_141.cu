
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for calculating mean along a dimension
__global__ void mean_kernel(const float* input, float* output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < 1) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += input[row * n + i];
        }
        output[row] = sum / n;
    }
}

// CUDA kernel for calculating determinant of a 2x2 matrix
__global__ void det2x2_kernel(const float* input, float* output, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        float a = input[idx * 4];
        float b = input[idx * 4 + 1];
        float c = input[idx * 4 + 2];
        float d = input[idx * 4 + 3];
        output[idx] = a * d - b * c;
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

    // Extract output tensors (assuming they're preallocated)
    float* output_mean = va_arg(args, float*);
    float* output_det = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int matrix_size = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output_mean, *d_output_det;
    cudaMalloc(&d_input, batch_size * matrix_size * sizeof(float));
    cudaMalloc(&d_output_mean, batch_size * sizeof(float));
    cudaMalloc(&d_output_det, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * matrix_size * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate mean
    dim3 threadsPerBlock(16, 1);
    dim3 numBlocks((1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    mean_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output_mean, batch_size, matrix_size);

    // Calculate determinant (assuming 2x2 matrices)
    threadsPerBlock = dim3(32, 1);
    numBlocks = dim3((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);
    det2x2_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output_det, batch_size);

    // Copy results back to host
    cudaMemcpy(output_mean, d_output_mean, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_det, d_output_det, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output_mean);
    cudaFree(d_output_det);
}

}  // extern "C"
