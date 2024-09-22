
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void roberts_mish_sum_kernel(const float* input, float* output, int batch_size, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width - 1 && y < height - 1 && b < batch_size) {
        // Roberts cross-gradient
        float dx = input[b * height * width + (y + 1) * width + x] - input[b * height * width + y * width + x];
        float dy = input[b * height * width + y * width + (x + 1)] - input[b * height * width + y * width + x];

        // Mish activation
        float mish = dx * tanhf(logf(expf(dx) + 1.0f)) + dy * tanhf(logf(expf(dy) + 1.0f));

        // Atomic add for sum reduction
        atomicAdd(output, mish);
    }
}

extern "C" {

void roberts_mish_sum(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int height = va_arg(args, int);
    int width = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * height * width * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float)); // Initialize output to 0

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    roberts_mish_sum_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, batch_size, height, width);

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}
