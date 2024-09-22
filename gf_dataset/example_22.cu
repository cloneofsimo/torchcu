
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>  // For fmaxf
#include <stdarg.h>  // For va_list, va_start, va_end

__global__ void noisy_relu_inplace_kernel(float* input, float noise_scale, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        input[i] += noise_scale * curand_uniform();
        input[i] = fmaxf(input[i], 0.0f); 
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    float* input = va_arg(args, float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);

    // Extract noise scale
    float noise_scale = va_arg(args, float);

    va_end(args);

    int size = input_dim0 * input_dim1;

    // Allocate device memory
    float *d_input;
    cudaMalloc(&d_input, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    noisy_relu_inplace_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, noise_scale, size
    );

    // Copy result back to host (inplace, so just overwrite)
    cudaMemcpy(input, d_input, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
}

} // extern "C"
