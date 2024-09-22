
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for element-wise multiplication and ReLU
__global__ void elementwise_mul_relu_kernel(const float* input_tensor, float scalar, float* output, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num_elements) {
        output[i] = fmaxf(input_tensor[i] * scalar, 0.0f);  // ReLU activation
    }
}

extern "C" {

void my_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract scalar
    float scalar = va_arg(args, double);  // Extract as double and then cast to float

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int num_elements = input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, num_elements * sizeof(float));
    cudaMalloc(&d_output, num_elements * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((num_elements + threadsPerBlock.x - 1) / threadsPerBlock.x);

    elementwise_mul_relu_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, scalar, d_output, num_elements
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
