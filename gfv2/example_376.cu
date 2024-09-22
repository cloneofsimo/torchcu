
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

__global__ void my_function_kernel(const float* input_tensor, const float* weights, float* output,
                                  float regularization_param, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += input_tensor[row * k + i] * weights[col * k + i];
        }
        output[row * n + col] = sum; // Intermediate result for einsum
    }
}

__global__ void activation_kernel(float* output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float val = output[row * n + col];
        output[row * n + col] = val * (1 + fabsf(val)); // Custom activation
    }
}

__global__ void regularization_kernel(float* output, const float* weights, float regularization_param, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float reg = 0.0f;
        for (int i = 0; i < k; ++i) {
            reg += fabsf(weights[col * k + i]);
        }
        output[row * n + col] -= regularization_param * reg;
    }
}


extern "C" {

void my_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    const float* weights = va_arg(args, const float*);
    int weights_dim0 = va_arg(args, int);
    int weights_dim1 = va_arg(args, int);

    float regularization_param = va_arg(args, float);

    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weights_dim0;

    // Allocate device memory
    float *d_input, *d_weights, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weights, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel for einsum
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    my_function_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weights, d_output, regularization_param, batch_size, output_dim, input_dim
    );

    // Launch kernel for activation
    activation_kernel<<<numBlocks, threadsPerBlock>>>(
        d_output, batch_size, output_dim
    );

    // Launch kernel for regularization
    regularization_kernel<<<numBlocks, threadsPerBlock>>>(
        d_output, d_weights, regularization_param, batch_size, output_dim, input_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
}

}  // extern "C"
