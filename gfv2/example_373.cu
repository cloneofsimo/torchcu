
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for the complex function
__global__ void my_complex_function_kernel(const float* input_tensor, const float scale, float* output, 
                                          int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        // Scale the input
        float scaled_input = input_tensor[row * n + col] * scale;

        // Apply sigmoid activation
        float activated_input = 1.0f / (1.0f + expf(-scaled_input));

        // Generate a random number
        float random_value = __float2int_rn(rand()) / (float)RAND_MAX; 

        // Element-wise multiplication
        output[row * n + col] = activated_input * random_value;
    }
}

// CUDA kernel for calculating row means
__global__ void row_mean_kernel(const float* output, float* row_means, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m) {
        float sum = 0.0f;
        for (int col = 0; col < n; ++col) {
            sum += output[row * n + col];
        }
        row_means[row] = sum / (float)n;
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

    // Extract scale
    float scale = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output, *d_row_means;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_row_means, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel for the complex function
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    my_complex_function_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, scale, d_output, batch_size, input_dim
    );

    // Launch kernel for calculating row means
    numBlocks = (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y;
    row_mean_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_row_means, batch_size, input_dim);

    // Copy result back to host
    cudaMemcpy(output, d_row_means, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_row_means);
}

} // extern "C"
