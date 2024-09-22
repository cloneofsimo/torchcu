
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for adaptive average pooling and sigmoid activation
__global__ void adaptive_avg_pool_sigmoid_kernel(const float* input_tensor, float* output,
                                                   int batch_size, int channels, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < channels) {
        float sum = 0.0f;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                sum += input_tensor[(row * channels + col) * height * width + i * width + j];
            }
        }
        output[row * channels + col] = 1.0f / (height * width) * sum;  // Average pooling
        output[row * channels + col] = 1.0f / (1.0f + expf(-output[row * channels + col])); // Sigmoid
    }
}

extern "C" {

void my_complex_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 8);
    dim3 numBlocks((channels + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    adaptive_avg_pool_sigmoid_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, channels, height, width
    );

    // Multiply by scalar on device
    for (int i = 0; i < batch_size * channels; i++) {
        d_output[i] *= 2.5f;
    }

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
