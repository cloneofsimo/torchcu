
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

extern "C" {

__global__ void cumsum_kernel(const int8_t* input, int8_t* output, int batch_size, int channels, int height, int width) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int h = threadIdx.z;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < batch_size && c < channels && h < height && w < width) {
        int idx = (b * channels + c) * height * width + h * width + w;
        int sum = 0;
        for (int i = 0; i <= h; ++i) {
            sum += input[(b * channels + c) * height * width + i * width + w];
        }
        output[idx] = sum;
    }
}

__global__ void conv2d_kernel(const int8_t* input, const float* kernel, int8_t* output, 
                              int batch_size, int in_channels, int out_channels, int height, int width, 
                              int kernel_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int h = threadIdx.z;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < batch_size && c < out_channels && h < height && w < width) {
        float sum = 0.0f;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int input_h = h + kh - kernel_size / 2;
                int input_w = w + kw - kernel_size / 2;

                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        int input_idx = (b * in_channels + ic) * height * width + input_h * width + input_w;
                        int kernel_idx = (c * in_channels + ic) * kernel_size * kernel_size + kh * kernel_size + kw;
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        output[(b * out_channels + c) * height * width + h * width + w] = (int8_t)sum;
    }
}

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const int8_t* input_tensor = va_arg(args, const int8_t*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    const float* kernel = va_arg(args, const float*);
    int kernel_dim0 = va_arg(args, int);
    int kernel_dim1 = va_arg(args, int);
    int kernel_dim2 = va_arg(args, int);
    int kernel_dim3 = va_arg(args, int);

    int8_t* output = va_arg(args, int8_t*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;
    int out_channels = kernel_dim1;
    int kernel_size = kernel_dim2; 

    // Allocate device memory
    int8_t *d_input, *d_output;
    float *d_kernel;
    cudaMalloc(&d_input, batch_size * in_channels * height * width * sizeof(int8_t));
    cudaMalloc(&d_output, batch_size * out_channels * height * width * sizeof(int8_t));
    cudaMalloc(&d_kernel, kernel_dim0 * kernel_dim1 * kernel_dim2 * kernel_dim3 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * in_channels * height * width * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_dim0 * kernel_dim1 * kernel_dim2 * kernel_dim3 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch cumsum kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                    (in_channels + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                    (height + threadsPerBlock.z - 1) / threadsPerBlock.z);
    cumsum_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, batch_size, in_channels, height, width);

    // Launch conv2d kernel
    numBlocks = ((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (out_channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (width + threadsPerBlock.z - 1) / threadsPerBlock.z);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_kernel, d_output, batch_size, in_channels, out_channels, height, width, kernel_size);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * out_channels * height * width * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

}  // extern "C"
