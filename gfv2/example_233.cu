
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

// Helper function for generating Gumbel noise
__device__ float gumbel_noise(float x) {
    return -log(-log(x));
}

// CUDA kernel for Gumbel-Softmax
__global__ void gumbel_softmax_kernel(const float* input_tensor, float temperature, 
                                     bool hard, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float log_prob = input_tensor[idx];
        float noise = gumbel_noise(rand() / (float)RAND_MAX); // Generate Gumbel noise
        float prob = exp((log_prob + noise) / temperature); // Calculate probability

        if (hard) {
            // One-hot encoding for hard Gumbel-Softmax
            output[idx] = (prob > 0.5f) ? 1.0f : 0.0f;
        } else {
            // Soft Gumbel-Softmax
            output[idx] = prob;
        }
    }
}

extern "C" {

void gumbel_softmax_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);

    // Extract temperature
    float temperature = va_arg(args, float);

    // Extract hard flag
    bool hard = va_arg(args, bool);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int num_elements = input_tensor_dim0;

    // Launch kernel
    dim3 threadsPerBlock(1024);
    dim3 numBlocks((num_elements + threadsPerBlock.x - 1) / threadsPerBlock.x);
    gumbel_softmax_kernel<<<numBlocks, threadsPerBlock>>>(input_tensor, temperature, hard, output, num_elements);

    cudaDeviceSynchronize();
}

} // extern "C"

