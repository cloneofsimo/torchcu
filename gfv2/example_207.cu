
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for multinomial sampling and minimum finding
__global__ void multinomial_min_kernel(const float* input_tensor, const float* probabilities, 
                                      int* samples, float* min_values, int num_samples, int data_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data_size) {
        // Multinomial sampling
        float random_value = __int2float(curand_uniform());
        float cumulative_prob = 0.0f;
        for (int i = 0; i < num_samples; ++i) {
            cumulative_prob += probabilities[i];
            if (random_value <= cumulative_prob) {
                samples[idx] = i;
                break;
            }
        }

        // Finding minimum value
        min_values[idx] = input_tensor[idx * num_samples + samples[idx]];
    }
}

extern "C" {

void multinomial_min_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);

    // Extract num_samples
    int num_samples = va_arg(args, int);

    // Extract probabilities tensor
    const float* probabilities = va_arg(args, const float*);
    int probabilities_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int data_size = input_tensor_dim0;

    // Allocate device memory
    float *d_input, *d_probabilities;
    int *d_samples;
    float *d_output;
    cudaMalloc(&d_input, data_size * num_samples * sizeof(float));
    cudaMalloc(&d_probabilities, num_samples * sizeof(float));
    cudaMalloc(&d_samples, data_size * sizeof(int));
    cudaMalloc(&d_output, data_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, data_size * num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probabilities, probabilities, num_samples * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (data_size + threadsPerBlock - 1) / threadsPerBlock;
    multinomial_min_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_probabilities, 
                                                        d_samples, d_output, num_samples, data_size);

    // Copy result back to host
    cudaMemcpy(output, d_output, data_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_probabilities);
    cudaFree(d_samples);
    cudaFree(d_output);
}

} // extern "C"
