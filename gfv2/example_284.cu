
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void gradient_clipping_kernel(float* input_tensor, float clip_value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (input_tensor[idx] > clip_value) {
            input_tensor[idx] = clip_value;
        } else if (input_tensor[idx] < -clip_value) {
            input_tensor[idx] = -clip_value;
        }
    }
}

extern "C" {

void gradient_clipping_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);

    // Extract clip value
    float clip_value = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int size = input_tensor_dim0;

    // Allocate device memory
    float *d_input;
    cudaMalloc(&d_input, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    gradient_clipping_kernel<<<numBlocks, threadsPerBlock>>>(d_input, clip_value, size);

    // Copy result back to host
    cudaMemcpy(output, d_input, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
}

}
