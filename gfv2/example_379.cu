
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for average pooling with kernel size 2x2 and stride 2
__global__ void avg_pool2d_kernel(const float* input, float* output, 
                                        int batch_size, int channels, 
                                        int input_height, int input_width, 
                                        int output_height, int output_width) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int channel = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && channel < channels && row < output_height) {
        int col = threadIdx.x; 
        float sum = 0.0f;

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int input_row = row * 2 + i;
                int input_col = col * 2 + j;

                if (input_row < input_height && input_col < input_width) {
                    sum += input[(batch * channels * input_height * input_width) + 
                                (channel * input_height * input_width) + 
                                (input_row * input_width) + input_col];
                }
            }
        }

        output[(batch * channels * output_height * output_width) + 
               (channel * output_height * output_width) + 
               (row * output_width) + col] = sum / 4.0f;
    }
}

// CUDA kernel for calculating the first singular value of SVD
__global__ void svd_kernel(const float* input, float* singular_values, 
                        int batch_size, int channels, int height, int width) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int channel = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && channel < channels && row == 0) {
        // Calculate sum of squares for each channel
        float sum = 0.0f;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                sum += input[(batch * channels * height * width) + 
                            (channel * height * width) + 
                            (i * width) + j] * 
                            input[(batch * channels * height * width) + 
                            (channel * height * width) + 
                            (i * width) + j];
            }
        }

        singular_values[(batch * channels) + channel] = sqrtf(sum);
    }
}

extern "C" {

void my_complex_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int channels = va_arg(args, int);
    int height = va_arg(args, int);
    int width = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int output_height = height / 2;
    int output_width = width / 2;

    // Allocate device memory
    float *d_input, *d_pooled, *d_singular_values;
    cudaMalloc(&d_input, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&d_pooled, batch_size * channels * output_height * output_width * sizeof(float));
    cudaMalloc(&d_singular_values, batch_size * channels * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch average pooling kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((output_width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (channels + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    avg_pool2d_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_pooled, batch_size, channels, 
                                                height, width, output_height, output_width);

    // Launch SVD kernel
    threadsPerBlock = dim3(1, 16, 1);
    numBlocks = dim3(1, (channels + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                    (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    svd_kernel<<<numBlocks, threadsPerBlock>>>(d_pooled, d_singular_values, 
                                    batch_size, channels, output_height, output_width);

    // Copy result back to host
    cudaMemcpy(output, d_singular_values, batch_size * channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_pooled);
    cudaFree(d_singular_values);
}

}  // extern "C"
