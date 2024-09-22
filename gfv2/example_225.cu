
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for adaptive max pooling on int8 data
__global__ void adaptive_max_pool_int8_kernel(const int8_t* input, float* output, int batch_size, int in_height, int in_width, int out_height, int out_width, int channels) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && c < channels) {
        int out_x = (c % out_width) * in_width / out_width;
        int out_y = (c / out_width) * in_height / out_height;

        int8_t max_value = input[b * in_height * in_width * channels + out_y * in_width * channels + out_x * channels + c];
        for (int y = out_y * in_height / out_height; y < (out_y + 1) * in_height / out_height; ++y) {
            for (int x = out_x * in_width / out_width; x < (out_x + 1) * in_width / out_width; ++x) {
                max_value = max(max_value, input[b * in_height * in_width * channels + y * in_width * channels + x * channels + c]);
            }
        }
        output[b * out_height * out_width * channels + out_y * out_width * channels + out_x * channels + c] = (float)max_value;
    }
}

extern "C" {

void adaptive_max_pool_int8_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const int8_t* input = va_arg(args, const int8_t*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);
    int input_dim2 = va_arg(args, int);
    int input_dim3 = va_arg(args, int);

    // Extract output size
    int output_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_dim0;
    int in_height = input_dim1;
    int in_width = input_dim2;
    int channels = input_dim3;
    int out_height = output_size;
    int out_width = output_size;

    // Allocate device memory
    int8_t *d_input;
    float *d_output;
    cudaMalloc(&d_input, batch_size * in_height * in_width * channels * sizeof(int8_t));
    cudaMalloc(&d_output, batch_size * out_height * out_width * channels * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * in_height * in_width * channels * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (channels + threadsPerBlock.y - 1) / threadsPerBlock.y);

    adaptive_max_pool_int8_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, batch_size, in_height, in_width, out_height, out_width, channels);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * out_height * out_width * channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
