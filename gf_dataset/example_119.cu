
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for calculating variance and its gradient
__global__ void var_backward_kernel(const float* input_tensor, float* grad_output,
                                    int num_elements, float mean) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        grad_output[idx] = 2.0f * (input_tensor[idx] - mean) / num_elements; 
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

    // Extract output tensor (assuming it's preallocated)
    float* grad_output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int num_elements = batch_size * input_dim;

    // Allocate device memory
    float *d_input, *d_grad_output;
    cudaMalloc(&d_input, num_elements * sizeof(float));
    cudaMalloc(&d_grad_output, num_elements * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate mean on the device (using a separate kernel or a reduction operation)
    float mean;
    // ... (Implementation for calculating the mean on the device)

    // Launch kernel for variance backward
    dim3 threadsPerBlock(256);
    dim3 numBlocks((num_elements + threadsPerBlock.x - 1) / threadsPerBlock.x);
    var_backward_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_grad_output,
                                                        num_elements, mean);

    // Copy result back to host
    cudaMemcpy(grad_output, d_grad_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_grad_output);
}

}  // extern "C"
