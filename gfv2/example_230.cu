
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void my_function_kernel(const float* input, float* output, int size, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Set the random seed for each thread
        curandState_t state;
        curand_init(seed + idx, 0, 0, &state);

        // Generate a random number and add it to the input
        float rand_num = curand_uniform(&state);
        output[idx] = input[idx] + 2.0f + rand_num;

        // Multiply by a scalar and apply sigmoid
        output[idx] *= 1.5f;
        output[idx] = 1.0f / (1.0f + exp(-output[idx]));
    }
}

extern "C" {

void my_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input = va_arg(args, const float*);
    int input_size = va_arg(args, int);
    int seed = va_arg(args, int);

    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((input_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    my_function_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, input_size, seed
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}
