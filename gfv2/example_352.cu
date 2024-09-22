
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for 3D transposed convolution with ReLU
__global__ void transposed_conv3d_relu_kernel(const float* input_tensor, const float* weight, const float* bias, float* output, 
                                            int batch_size, int in_channels, int in_depth, int in_height, int in_width,
                                            int out_channels, int kernel_depth, int kernel_height, int kernel_width,
                                            int stride_depth, int stride_height, int stride_width,
                                            int padding_depth, int padding_height, int padding_width,
                                            int output_padding_depth, int output_padding_height, int output_padding_width) {
    
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int out_channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_depth_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_channel_idx < out_channels && out_depth_idx < in_depth + 2 * padding_depth + output_padding_depth) {
        
        int out_height_idx = 0;
        int out_width_idx = 0;

        for (int kernel_z = 0; kernel_z < kernel_depth; ++kernel_z) {
            for (int kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
                for (int kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                    
                    int in_depth_idx = out_depth_idx - kernel_z * stride_depth + padding_depth;
                    int in_height_idx = out_height_idx - kernel_y * stride_height + padding_height;
                    int in_width_idx = out_width_idx - kernel_x * stride_width + padding_width;

                    if (in_depth_idx >= 0 && in_depth_idx < in_depth &&
                        in_height_idx >= 0 && in_height_idx < in_height &&
                        in_width_idx >= 0 && in_width_idx < in_width) {

                        for (int in_channel_idx = 0; in_channel_idx < in_channels; ++in_channel_idx) {

                            int input_index = batch_idx * in_channels * in_depth * in_height * in_width + 
                                              in_channel_idx * in_depth * in_height * in_width + 
                                              in_depth_idx * in_height * in_width + 
                                              in_height_idx * in_width + 
                                              in_width_idx;

                            int weight_index = out_channel_idx * in_channels * kernel_depth * kernel_height * kernel_width + 
                                              in_channel_idx * kernel_depth * kernel_height * kernel_width + 
                                              kernel_z * kernel_height * kernel_width + 
                                              kernel_y * kernel_width + 
                                              kernel_x;

                            output[batch_idx * out_channels * (in_depth + 2 * padding_depth + output_padding_depth) *
                                   (in_height + 2 * padding_height + output_padding_height) * 
                                   (in_width + 2 * padding_width + output_padding_width) + 
                                   out_channel_idx * (in_depth + 2 * padding_depth + output_padding_depth) * 
                                   (in_height + 2 * padding_height + output_padding_height) * 
                                   (in_width + 2 * padding_width + output_padding_width) + 
                                   out_depth_idx * (in_height + 2 * padding_height + output_padding_height) * 
                                   (in_width + 2 * padding_width + output_padding_width) + 
                                   out_height_idx * (in_width + 2 * padding_width + output_padding_width) + 
                                   out_width_idx] += input_tensor[input_index] * weight[weight_index];

                        }
                    }
                }
            }
        }
        output[batch_idx * out_channels * (in_depth + 2 * padding_depth + output_padding_depth) * 
               (in_height + 2 * padding_height + output_padding_height) * 
               (in_width + 2 * padding_width + output_padding_width) + 
               out_channel_idx * (in_depth + 2 * padding_depth + output_padding_depth) * 
               (in_height + 2 * padding_height + output_padding_height) * 
               (in_width + 2 * padding_width + output_padding_width) + 
               out_depth_idx * (in_height + 2 * padding_height + output_padding_height) * 
               (in_width + 2 * padding_width + output_padding_width) + 
               out_height_idx * (in_width + 2 * padding_width + output_padding_width) + 
               out_width_idx] += bias[out_channel_idx];
        output[batch_idx * out_channels * (in_depth + 2 * padding_depth + output_padding_depth) * 
               (in_height + 2 * padding_height + output_padding_height) * 
               (in_width + 2 * padding_width + output_padding_width) + 
               out_channel_idx * (in_depth + 2 * padding_depth + output_padding_depth) * 
               (in_height + 2 * padding_height + output_padding_height) * 
               (in_width + 2 * padding_width + output_padding_width) + 
               out_depth_idx * (in_height + 2 * padding_height + output_padding_height) * 
               (in_width + 2 * padding_width + output_padding_width) + 
               out_height_idx * (in_width + 2 * padding_width + output_padding_width) + 
               out_width_idx] = fmaxf(output[batch_idx * out_channels * (in_depth + 2 * padding_depth + output_padding_depth) * 
                                        (in_height + 2 * padding_height + output_padding_height) * 
                                        (in_width + 2 * padding_width + output_padding_width) + 
                                        out_channel_idx * (in_depth + 2 * padding_depth + output_padding_depth) * 
                                        (in_height + 2 * padding_height + output_padding_height) * 
                                        (in_width + 2 * padding_width + output_padding_width) + 
                                        out_depth_idx * (in_height + 2 * padding_height + output_padding_height) * 
                                        (in_width + 2 * padding_width + output_padding_width) + 
                                        out_height_idx * (in_width + 2 * padding_width + output_padding_width) + 
                                        out_width_idx], 0.0f); 
    }
}

extern "C" {

void transposed_conv3d_example(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);
    int input_tensor_dim4 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);
    int weight_dim3 = va_arg(args, int);
    int weight_dim4 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int in_depth = input_tensor_dim2;
    int in_height = input_tensor_dim3;
    int in_width = input_tensor_dim4;

    int out_channels = weight_dim0;
    int kernel_depth = weight_dim2;
    int kernel_height = weight_dim3;
    int kernel_width = weight_dim4;

    int stride_depth = 2;
    int stride_height = 2;
    int stride_width = 2;

    int padding_depth = 1;
    int padding_height = 1;
    int padding_width = 1;

    int output_padding_depth = 1;
    int output_padding_height = 1;
    int output_padding_width = 1;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * in_channels * in_depth * in_height * in_width * sizeof(float));
    cudaMalloc(&d_weight, out_channels * in_channels * kernel_depth * kernel_height * kernel_width * sizeof(float));
    cudaMalloc(&d_bias, out_channels * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * (in_depth + 2 * padding_depth + output_padding_depth) * 
                              (in_height + 2 * padding_height + output_padding_height) * 
                              (in_width + 2 * padding_width + output_padding_width) * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * in_channels * in_depth * in_height * in_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, out_channels * in_channels * kernel_depth * kernel_height * kernel_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((in_depth + 2 * padding_depth + output_padding_depth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (out_channels + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    transposed_conv3d_relu_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output,
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_depth, kernel_height, kernel_width,
        stride_depth, stride_height, stride_width,
        padding_depth, padding_height, padding_width,
        output_padding_depth, output_padding_height, output_padding_width
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * out_channels * (in_depth + 2 * padding_depth + output_padding_depth) * 
                              (in_height + 2 * padding_height + output_padding_height) * 
                              (in_width + 2 * padding_width + output_padding_width) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
