
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for PReLU backward pass
__global__ void prelu_backward_kernel(const float* input_tensor, const float* weight, float* grad_input, 
                                        int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m * n) {
        if (input_tensor[idx] > 0.0f) {
            grad_input[idx] = 1.0f; 
        } else {
            grad_input[idx] = weight[0];  // Access the single weight value
        }
    }
}

extern "C" {

void prelu_backward_example(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);  // Assuming weight is a single value

    // Extract output tensor (assuming it's preallocated)
    float* grad_input = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory for input and output
    float *d_input, *d_weight, *d_grad_input;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * sizeof(float));  // Allocate for single weight
    cudaMalloc(&d_grad_input, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size * input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x);

    prelu_backward_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_grad_input, 
                                                        batch_size * input_dim, 1); // Pass 1 for weight dimension

    // Copy result back to host
    cudaMemcpy(grad_input, d_grad_input, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_grad_input);
}

}  // extern "C"
