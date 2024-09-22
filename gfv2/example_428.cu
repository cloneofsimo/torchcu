
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

#define THREADS_PER_BLOCK 16

__global__ void matmul_sigmoid_kernel(const float* input_tensor, const float* weight, float* output, 
                                        int m, int n, int k, int pad_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            int input_idx = (row * k + i) + pad_size + (row * pad_size) + (i * pad_size);
            sum += input_tensor[input_idx] * weight[col * k + i];
        }
        output[row * n + col] = 1.0f / (1.0f + expf(-sum)); // Sigmoid activation
    }
}

__global__ void gradient_kernel(const float* output_grad, const float* weight, float* input_grad,
                                 int m, int n, int k, int pad_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            int input_idx = (row * k + i) + pad_size + (row * pad_size) + (i * pad_size);
            sum += output_grad[row * n + i] * weight[col * k + i]; 
        }
        input_grad[input_idx] = sum; // Accumulate gradient for each element
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
    int input_tensor_dim2 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);
    float* input_grad = va_arg(args, float*); // For gradient

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;
    int pad_size = 2; // Padding size

    // Allocate device memory
    float *d_input, *d_weight, *d_output, *d_input_grad, *d_output_grad;
    cudaMalloc(&d_input, (batch_size * input_dim * input_dim + (pad_size * input_dim * input_dim)) * sizeof(float));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));
    cudaMalloc(&d_input_grad, (batch_size * input_dim * input_dim + (pad_size * input_dim * input_dim)) * sizeof(float));
    cudaMalloc(&d_output_grad, batch_size * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel for matmul and sigmoid
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_sigmoid_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, batch_size, output_dim, input_dim, pad_size
    );

    // Launch kernel for gradient calculation
    dim3 numBlocks_grad((input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaMemset(d_input_grad, 0, (batch_size * input_dim * input_dim + (pad_size * input_dim * input_dim)) * sizeof(float)); // Initialize gradient
    cudaMemcpy(d_output_grad, output, batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice);
    
    gradient_kernel<<<numBlocks_grad, threadsPerBlock>>>(d_output_grad, d_weight, d_input_grad,
                                                       batch_size, input_dim, output_dim, pad_size);

    // Copy results back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(input_grad, d_input_grad, (batch_size * input_dim * input_dim + (pad_size * input_dim * input_dim)) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaFree(d_input_grad);
    cudaFree(d_output_grad);
}
} // extern "C"
