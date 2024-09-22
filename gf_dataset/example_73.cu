
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for Hadamard product and sigmoid
__global__ void hadamard_sigmoid_kernel(const float* input_tensor, const float* weight, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __expf(-__fmaf(-input_tensor[idx] * weight[idx], 1.0f, 0.0f)) / (1.0f + __expf(-__fmaf(-input_tensor[idx] * weight[idx], 1.0f, 0.0f)));
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

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int size = batch_size * input_dim;
    dim3 threadsPerBlock(1024);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    hadamard_sigmoid_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, size
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
