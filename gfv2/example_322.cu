
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function for gradient clipping
__device__ float clip(float x, float min, float max) {
    return fminf(fmaxf(x, min), max);
}

__global__ void depthwise_separable_conv_kernel(const float* input, const float* depthwise_weight, const float* pointwise_weight, const float* bias, float* output, 
                                                int batch_size, int input_channels, int input_height, int input_width, 
                                                int output_channels, int kernel_height, int kernel_width, int padding,
                                                int stride) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (b < batch_size && y < input_height && x < input_width) {
        float sum = 0.0f;
        
        // Depthwise convolution
        for (int c = 0; c < input_channels; c++) {
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    int input_y = y + ky - padding;
                    int input_x = x + kx - padding;
                    if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                        sum += input[b * input_channels * input_height * input_width + c * input_height * input_width + input_y * input_width + input_x] *
                               depthwise_weight[c * kernel_height * kernel_width + ky * kernel_width + kx];
                    }
                }
            }
        }
        
        // Pointwise convolution
        for (int oc = 0; oc < output_channels; oc++) {
            sum = sum * pointwise_weight[oc * input_channels + c];
            sum += bias[oc];
            
            // ReLU activation
            sum = fmaxf(sum, 0.0f);
            
            // Gradient clipping
            sum = clip(sum, -1.0f, 1.0f);
            
            output[b * output_channels * input_height * input_width + oc * input_height * input_width + y * input_width + x] = sum;
        }
    }
}

extern "C" {

void depthwise_separable_conv_with_clipping(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int input_channels = va_arg(args, int);
    int input_height = va_arg(args, int);
    int input_width = va_arg(args, int);

    // Extract depthwise weight tensor
    const float* depthwise_weight = va_arg(args, const float*);
    int depthwise_channels = va_arg(args, int);
    int kernel_height = va_arg(args, int);
    int kernel_width = va_arg(args, int);

    // Extract pointwise weight tensor
    const float* pointwise_weight = va_arg(args, const float*);
    int output_channels = va_arg(args, int);

    // Extract bias tensor (optional)
    const float* bias = va_arg(args, const float*);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Calculate output dimensions (assuming same padding and stride 1)
    int padding = (kernel_height - 1) / 2; // Same padding
    int stride = 1;
    int output_height = input_height;
    int output_width = input_width;

    // Allocate device memory
    float *d_input, *d_depthwise_weight, *d_pointwise_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * input_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_depthwise_weight, depthwise_channels * kernel_height * kernel_width * sizeof(float));
    cudaMalloc(&d_pointwise_weight, output_channels * input_channels * sizeof(float));
    if (bias) {
        cudaMalloc(&d_bias, output_channels * sizeof(float));
        cudaMemcpy(d_bias, bias, output_channels * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&d_output, batch_size * output_channels * output_height * output_width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * input_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_depthwise_weight, depthwise_weight, depthwise_channels * kernel_height * kernel_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointwise_weight, pointwise_weight, output_channels * input_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (output_width + threadsPerBlock.z - 1) / threadsPerBlock.z);

    depthwise_separable_conv_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_depthwise_weight, d_pointwise_weight, d_bias, d_output, 
        batch_size, input_channels, input_height, input_width, 
        output_channels, kernel_height, kernel_width, padding, stride
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_channels * output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_depthwise_weight);
    cudaFree(d_pointwise_weight);
    if (bias) {
        cudaFree(d_bias);
    }
    cudaFree(d_output);
}

}  // extern "C"
