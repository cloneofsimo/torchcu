
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for permute, unfold, and convolution
__global__ void conv_permute_unfold_kernel(const float* input_tensor, const float* weight, const float* bias, 
                                        float* output, int batch_size, int in_channels, int in_height, int in_width,
                                        int out_channels, int kernel_height, int kernel_width) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;
    int o = threadIdx.w;

    if (b < batch_size && h < in_height && w < in_width && o < out_channels) {
        float sum = 0.0f;
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int in_idx = (b * in_channels * in_height * in_width) + (o * in_height * in_width) + 
                            ((h + kh) * in_width) + (w + kw);
                int weight_idx = (o * in_channels * kernel_height * kernel_width) + 
                                 (kh * kernel_width) + kw;
                sum += input_tensor[in_idx] * weight[weight_idx];
            }
        }

        // Add bias
        sum += bias[o];
        output[(b * out_channels * in_height * in_width) + (h * in_width * out_channels) + 
                 (w * out_channels) + o] = sum;
    }
}

extern "C" {

void conv_permute_unfold(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);
    int weight_dim3 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int in_height = input_tensor_dim2;
    int in_width = input_tensor_dim3;

    int out_channels = weight_dim0;
    int kernel_height = weight_dim2;
    int kernel_width = weight_dim3;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * in_channels * in_height * in_width * sizeof(float));
    cudaMalloc(&d_weight, out_channels * in_channels * kernel_height * kernel_width * sizeof(float));
    cudaMalloc(&d_bias, out_channels * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * in_height * in_width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * in_channels * in_height * in_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, out_channels * in_channels * kernel_height * kernel_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 16, 1);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (in_height + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                   (in_width + threadsPerBlock.z - 1) / threadsPerBlock.z);

    conv_permute_unfold_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_height, kernel_width
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
