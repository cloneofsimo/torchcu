
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 
#include <math.h>

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
        output[row * n + col] = 1.0f / (1.0f + expf(-sum));  // sigmoid activation
    }
}

// CUDA kernel for calculating NLL loss
__global__ void nll_loss_kernel(const float* output, const int* target, float* loss, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        loss[0] += -logf(output[idx * batch_size + target[idx]]);
    }
}

// CUDA kernel for calculating Frobenius norm
__global__ void frobenius_norm_kernel(const float* input_tensor, float* norm, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        atomicAdd(norm, input_tensor[row * n + col] * input_tensor[row * n + col]);
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

    // Extract target tensor
    const int* target_tensor = va_arg(args, const int*);
    int target_tensor_dim0 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    // Extract output tensors for nll_loss and frobenius_norm
    float* nll_loss = va_arg(args, float*);
    float* frobenius_norm = va_arg(args, float*);
    int8_t* ceiled_norm = va_arg(args, int8_t*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

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

    // Launch kernel for calculating NLL loss
    numBlocks = (batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    nll_loss_kernel<<<numBlocks, threadsPerBlock>>>(d_output, target_tensor, nll_loss, batch_size);

    // Launch kernel for calculating Frobenius norm
    numBlocks = ((input_tensor_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_tensor_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);
    frobenius_norm_kernel<<<numBlocks, threadsPerBlock>>>(d_input, frobenius_norm, input_tensor_dim0, input_tensor_dim1);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nll_loss, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(frobenius_norm, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate ceiled norm on host
    ceiled_norm[0] = (int8_t)ceilf(frobenius_norm[0]);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
