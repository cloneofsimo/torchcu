
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for convolution (TBC format) using int8
__global__ void conv_tbc_int8_kernel(const int8_t* input, const int8_t* weight, const int8_t* bias,
                                     float* output, int batch_size, int in_channels, int out_channels,
                                     int in_height, int in_width, int kernel_height, int kernel_width,
                                     int padding) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.z * blockDim.z + threadIdx.z;

    if (batch < batch_size && out_y < in_height && out_x < in_width) {
        float sum = 0.0f;
        for (int k = 0; k < in_channels; k++) {
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    int in_y = out_y + ky - padding;
                    int in_x = out_x + kx - padding;
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        sum += (float)(input[batch * in_channels * in_height * in_width + k * in_height * in_width + in_y * in_width + in_x] *
                                     weight[k * out_channels * kernel_height * kernel_width + ky * kernel_width + kx]);
                    }
                }
            }
        }
        sum += (float)bias[out_channels * batch + (out_y * in_width + out_x)];
        output[batch * out_channels * in_height * in_width + out_y * in_width + out_x] = sum;
    }
}

extern "C" {

void conv_tbc_int8_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const int8_t* input = va_arg(args, const int8_t*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);
    int input_dim2 = va_arg(args, int);
    int input_dim3 = va_arg(args, int);

    // Extract weight tensor
    const int8_t* weight = va_arg(args, const int8_t*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);
    int weight_dim3 = va_arg(args, int);

    // Extract bias tensor
    const int8_t* bias = va_arg(args, const int8_t*);
    int bias_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_dim0;
    int in_channels = input_dim1;
    int out_channels = weight_dim1;
    int in_height = input_dim2;
    int in_width = input_dim3;
    int kernel_height = weight_dim2;
    int kernel_width = weight_dim3;
    int padding = 1; // Assuming padding = 1

    // Allocate device memory
    int8_t *d_input, *d_weight, *d_bias;
    float *d_output;
    cudaMalloc(&d_input, batch_size * in_channels * in_height * in_width * sizeof(int8_t));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * sizeof(int8_t));
    cudaMalloc(&d_bias, bias_dim0 * sizeof(int8_t));
    cudaMalloc(&d_output, batch_size * out_channels * in_height * in_width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * in_channels * in_height * in_width * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_dim0 * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (in_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (in_width + threadsPerBlock.z - 1) / threadsPerBlock.z);

    conv_tbc_int8_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output, batch_size, in_channels, out_channels,
        in_height, in_width, kernel_height, kernel_width, padding
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * out_channels * in_height * in_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
