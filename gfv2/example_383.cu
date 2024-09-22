
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>  // For expf, logf
#include <stdarg.h> 

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* input_tensor, const float* weight, float* output,
                               int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += input_tensor[row * k + i] * weight[col * k + i];
        }
        output[row * n + col] = sum;
    }
}

// CUDA kernel for normalization (l2 normalization)
__global__ void normalize_kernel(float* data, int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {
        float sum_sq = 0.0f;
        for (int i = 0; i < num_cols; ++i) {
            sum_sq += data[row * num_cols + i] * data[row * num_cols + i];
        }
        float norm = rsqrtf(sum_sq);
        data[row * num_cols + col] *= norm;
    }
}

// CUDA kernel for cosine similarity calculation
__global__ void cosine_similarity_kernel(const float* output, const float* weights, int batch_size,
                                        int num_cols, const int* labels, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        float dot_product = 0.0f;
        for (int i = 0; i < num_cols; ++i) {
            dot_product += output[idx * num_cols + i] * weights[labels[idx] * num_cols + i];
        }
        result[idx] = dot_product;
    }
}

extern "C" {

void my_complex_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    const float* weights = va_arg(args, const float*);
    int weights_dim0 = va_arg(args, int);
    int weights_dim1 = va_arg(args, int);

    const int* labels = va_arg(args, const int*);
    int labels_dim = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weights_dim0;

    // Allocate device memory
    float* d_input, *d_weights, *d_output, *d_result;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weights, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));
    cudaMalloc(&d_result, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // 1. Logarithm of the input tensor
    for (int i = 0; i < batch_size * input_dim; i++) {
        d_input[i] = logf(d_input[i]);
    }

    // 2. Matrix multiplication
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weights, d_output, batch_size, output_dim, input_dim
    );

    // 3. Arcface Loss Calculation
    // 3.1 Normalize output
    normalize_kernel<<<numBlocks, threadsPerBlock>>>(d_output, batch_size, output_dim);
    // 3.2 Normalize weights
    normalize_kernel<<<numBlocks, threadsPerBlock>>>(d_weights, output_dim, input_dim);
    // 3.3 Calculate cosine similarity
    cosine_similarity_kernel<<<(batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                                  threadsPerBlock>>>(d_output, d_weights, batch_size, output_dim, labels, d_result);

    // 4. In-place modification of the result
    for (int i = 0; i < batch_size * output_dim; i++) {
        d_output[i] *= d_result[i / output_dim];
    }

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaFree(d_result);
}

} // extern "C"
