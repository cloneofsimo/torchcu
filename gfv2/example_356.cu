
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for the complex function
__global__ void complex_function_kernel(const float* input_tensor, const float* weight1, const float* weight2, float* output,
                                        int batch_size, int input_dim, int hidden_dim1, int hidden_dim2) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < hidden_dim2) {
        float sum1 = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            sum1 += input_tensor[row * input_dim + i] * weight1[col * input_dim + i];
        }

        float sum2 = 0.0f;
        for (int i = 0; i < hidden_dim1; ++i) {
            sum2 += sum1 * weight2[col * hidden_dim1 + i];
        }

        output[row * hidden_dim2 + col] = sum1 + sum2;
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

    // Extract weight1 tensor
    const float* weight1 = va_arg(args, const float*);
    int weight1_dim0 = va_arg(args, int);
    int weight1_dim1 = va_arg(args, int);

    // Extract weight2 tensor
    const float* weight2 = va_arg(args, const float*);
    int weight2_dim0 = va_arg(args, int);
    int weight2_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int hidden_dim1 = weight1_dim0;
    int hidden_dim2 = weight2_dim0;

    // Allocate device memory
    float *d_input, *d_weight1, *d_weight2, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight1, hidden_dim1 * input_dim * sizeof(float));
    cudaMalloc(&d_weight2, hidden_dim2 * hidden_dim1 * sizeof(float));
    cudaMalloc(&d_output, batch_size * hidden_dim2 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight1, weight1, hidden_dim1 * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight2, weight2, hidden_dim2 * hidden_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 16);
    dim3 numBlocks((hidden_dim2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    complex_function_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight1, d_weight2, d_output, batch_size, input_dim, hidden_dim1, hidden_dim2
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * hidden_dim2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight1);
    cudaFree(d_weight2);
    cudaFree(d_output);
}

}  // extern "C"
