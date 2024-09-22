
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void spectral_rolloff_kernel(const float* input_tensor, float* output, int batch_size, int input_dim, float rolloff_freq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        float sum = 0.0f;
        for (int j = 0; j < input_dim; ++j) {
            sum += input_tensor[i * input_dim + j];
        }
        output[i] = sum / input_dim * rolloff_freq;
    }
}

__global__ void spectral_rolloff_backward_kernel(const float* grad_output, const float* input_tensor, float* grad_input, int batch_size, int input_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        for (int j = 0; j < input_dim; ++j) {
            grad_input[i * input_dim + j] = grad_output[i] / input_dim;
        }
    }
}

extern "C" {
void spectral_rolloff_module(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    // Extract rolloff frequency
    float rolloff_freq = va_arg(args, float);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    spectral_rolloff_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, batch_size, input_dim, rolloff_freq);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

void spectral_rolloff_module_backward(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract grad_output tensor
    const float* grad_output = va_arg(args, const float*);
    int grad_output_dim0 = va_arg(args, int);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract grad_input tensor
    float* grad_input = va_arg(args, float*);

    va_end(args);

    int batch_size = grad_output_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_grad_output, *d_input, *d_grad_input;
    cudaMalloc(&d_grad_output, batch_size * sizeof(float));
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_grad_input, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_grad_output, grad_output, batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    spectral_rolloff_backward_kernel<<<numBlocks, threadsPerBlock>>>(d_grad_output, d_input, d_grad_input, batch_size, input_dim);

    // Copy result back to host
    cudaMemcpy(grad_input, d_grad_input, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_grad_output);
    cudaFree(d_input);
    cudaFree(d_grad_input);
}

}  // extern "C"
