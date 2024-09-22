
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for matrix multiplication and sigmoid activation
__global__ void matmul_sigmoid_kernel(const float* input_tensor, const float* weight, float* output, 
                                        int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += input_tensor[row * k + i] * weight[col * k + i];  // Transposed access
        }
        output[row * n + col] = 1.0f / (1.0f + expf(-sum));  // Sigmoid activation
    }
}

// CUDA kernel for mean calculation along a specific dimension
__global__ void mean_kernel(const float* input, float* output, int m, int n, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += input[row * n + i];
        }
        output[row * n + col] = sum / n;
    }
}

extern "C" {

void my_complex_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract output tensors (assuming they're preallocated)
    float* output_mean = va_arg(args, float*);
    float* ones_tensor = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;

    // Allocate device memory
    float *d_input, *d_weight, *d_output, *d_output_mean, *d_ones;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));
    cudaMalloc(&d_output_mean, batch_size * output_dim * sizeof(float));
    cudaMalloc(&d_ones, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel for matrix multiplication and sigmoid activation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_sigmoid_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, batch_size, output_dim, input_dim
    );

    // Launch kernel for mean calculation
    dim3 mean_blocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    mean_kernel<<<mean_blocks, threadsPerBlock>>>(d_output, d_output_mean, batch_size, output_dim, 1); 

    // Copy output_mean back to host
    cudaMemcpy(output_mean, d_output_mean, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Set ones tensor on device
    cudaMemset(d_ones, 1, batch_size * input_dim * sizeof(float));

    // Copy ones tensor back to host
    cudaMemcpy(ones_tensor, d_ones, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaFree(d_output_mean);
    cudaFree(d_ones);
}

}  // extern "C"
