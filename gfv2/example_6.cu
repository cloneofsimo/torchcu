
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for nearest neighbor upsampling
__global__ void upsample_nearest_kernel(const float* input, float* output,
                                         int batch_size, int channels,
                                         int input_height, int input_width,
                                         int output_height, int output_width,
                                         int scale_factor) {
    int b = blockIdx.z * blockDim.z + threadIdx.z;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && c < channels && h < output_height) {
        int in_h = h / scale_factor;
        int in_w = (h % scale_factor) * scale_factor;  // Handle horizontal upsampling
        int in_idx = (b * channels + c) * input_height * input_width + in_h * input_width + in_w;

        for (int w = 0; w < scale_factor; w++) {
            int out_idx = (b * channels + c) * output_height * output_width + h * output_width + w;
            output[out_idx] = input[in_idx];
        }
    }
}

extern "C" {

void upsample_nearest_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);
    int input_dim2 = va_arg(args, int);
    int input_dim3 = va_arg(args, int);

    // Extract scale factor
    int scale_factor = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_dim0;
    int channels = input_dim1;
    int input_height = input_dim2;
    int input_width = input_dim3;
    int output_height = input_height * scale_factor;
    int output_width = input_width * scale_factor;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * output_height * output_width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((output_height + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    upsample_nearest_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, channels,
        input_height, input_width, output_height, output_width, scale_factor
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
