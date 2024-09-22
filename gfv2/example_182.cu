
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

extern "C" {

__global__ void adversarial_training_kernel(const float* input_tensor, const float* weights, const float* perturbation, 
                                            float* output, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += (input_tensor[row * k + i] + perturbation[row * k + i]) * weights[col * k + i];  // Transposed access
        }
        output[row * n + col] = sum;
    }
}

void adversarial_training_example(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weights tensor
    const float* weights = va_arg(args, const float*);
    int weights_dim0 = va_arg(args, int);
    int weights_dim1 = va_arg(args, int);

    // Extract perturbation tensor
    const float* perturbation = va_arg(args, const float*);
    int perturbation_dim0 = va_arg(args, int);
    int perturbation_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weights_dim0;

    // Allocate device memory
    float *d_input, *d_weights, *d_perturbation, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weights, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_perturbation, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_perturbation, perturbation, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    adversarial_training_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weights, d_perturbation, d_output, batch_size, output_dim, input_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_perturbation);
    cudaFree(d_output);
}

}  // extern "C"
