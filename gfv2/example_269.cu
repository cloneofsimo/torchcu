
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function for fast int8 conversion
__device__ __forceinline__ int8_t float_to_int8(float f) {
    return static_cast<int8_t>(f);
}

// CUDA kernel for transposed 3D convolution with int8 output
__global__ void transposed_conv3d_int8_kernel(const float* input, const float* weight, const float* bias, 
                                           int8_t* output, int batch, int in_channels, int in_depth, int in_height, int in_width,
                                           int out_channels, int kernel_depth, int kernel_height, int kernel_width,
                                           int stride_depth, int stride_height, int stride_width,
                                           int padding_depth, int padding_height, int padding_width,
                                           int dilation_depth, int dilation_height, int dilation_width,
                                           int output_depth, int output_height, int output_width) {
    int b = blockIdx.x;
    int out_z = blockIdx.y * blockDim.y + threadIdx.y;
    int out_y = blockIdx.z * blockDim.z + threadIdx.z;
    int out_x = threadIdx.x;

    if (out_z < output_depth && out_y < output_height && out_x < output_width && b < batch) {
        float sum = bias[0];
        for (int k = 0; k < out_channels; ++k) {
            for (int i = 0; i < kernel_depth; ++i) {
                for (int j = 0; j < kernel_height; ++j) {
                    for (int l = 0; l < kernel_width; ++l) {
                        int in_z = out_z * stride_depth - padding_depth + i * dilation_depth;
                        int in_y = out_y * stride_height - padding_height + j * dilation_height;
                        int in_x = out_x * stride_width - padding_width + l * dilation_width;

                        if (in_z >= 0 && in_z < in_depth && in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                            sum += input[b * in_channels * in_depth * in_height * in_width + k * in_depth * in_height * in_width 
                                      + in_z * in_height * in_width + in_y * in_width + in_x] *
                                   weight[k * kernel_depth * kernel_height * kernel_width + i * kernel_height * kernel_width 
                                      + j * kernel_width + l];
                        }
                    }
                }
            }
            output[b * out_channels * output_depth * output_height * output_width + k * output_depth * output_height * output_width
                    + out_z * output_height * output_width + out_y * output_width + out_x] = float_to_int8(sum);
        }
    }
}

// CUDA kernel for calculating matrix rank
__global__ void matrix_rank_kernel(const int8_t* data, int* rank, int batch, int channels, int depth, int height, int width) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int d = blockIdx.z;
    int h = threadIdx.x;
    int w = threadIdx.y;

    if (h < width && w < height && b < batch && c < channels && d < depth) {
        int idx = b * channels * depth * height * width + c * depth * height * width + d * height * width + h * width + w;
        if (data[idx] != 0) {
            atomicAdd(rank, 1);
        }
    }
}

extern "C" {
    // Function to compute transposed 3D convolution and return int8 output and matrix rank
    void transposed_conv3d_rank_int8(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract arguments
        const float* input = va_arg(args, const float*);
        int input_dim0 = va_arg(args, int);
        int input_dim1 = va_arg(args, int);
        int input_dim2 = va_arg(args, int);
        int input_dim3 = va_arg(args, int);
        int input_dim4 = va_arg(args, int);

        const float* weight = va_arg(args, const float*);
        int weight_dim0 = va_arg(args, int);
        int weight_dim1 = va_arg(args, int);
        int weight_dim2 = va_arg(args, int);
        int weight_dim3 = va_arg(args, int);
        int weight_dim4 = va_arg(args, int);

        const float* bias = va_arg(args, const float*);
        int bias_dim0 = va_arg(args, int);

        int8_t* output = va_arg(args, int8_t*);
        int output_dim0 = va_arg(args, int);
        int output_dim1 = va_arg(args, int);
        int output_dim2 = va_arg(args, int);
        int output_dim3 = va_arg(args, int);
        int output_dim4 = va_arg(args, int);

        int* rank = va_arg(args, int*);

        va_end(args);

        // Calculate output dimensions
        int output_depth = (input_dim2 - 1) * weight_dim2 + weight_dim4 - 2 * weight_dim3 + 1;
        int output_height = (input_dim3 - 1) * weight_dim3 + weight_dim3 - 2 * weight_dim2 + 1;
        int output_width = (input_dim4 - 1) * weight_dim4 + weight_dim4 - 2 * weight_dim1 + 1;

        // Allocate device memory
        float* d_input, *d_weight, *d_bias;
        int8_t* d_output;
        int* d_rank;
        cudaMalloc(&d_input, input_dim0 * input_dim1 * input_dim2 * input_dim3 * input_dim4 * sizeof(float));
        cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * weight_dim4 * sizeof(float));
        cudaMalloc(&d_bias, bias_dim0 * sizeof(float));
        cudaMalloc(&d_output, output_dim0 * output_dim1 * output_dim2 * output_dim3 * output_dim4 * sizeof(int8_t));
        cudaMalloc(&d_rank, sizeof(int));

        // Copy data to device
        cudaMemcpy(d_input, input, input_dim0 * input_dim1 * input_dim2 * input_dim3 * input_dim4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * weight_dim4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, bias, bias_dim0 * sizeof(float), cudaMemcpyHostToDevice);

        // Launch transposed convolution kernel
        dim3 threadsPerBlock(output_width, 1, 1);
        dim3 numBlocks(input_dim0, (output_depth + threadsPerBlock.y - 1) / threadsPerBlock.y, (output_height + threadsPerBlock.z - 1) / threadsPerBlock.z);

        transposed_conv3d_int8_kernel<<<numBlocks, threadsPerBlock>>>(
            d_input, d_weight, d_bias, d_output, input_dim0, input_dim1, input_dim2, input_dim3, input_dim4,
            weight_dim0, weight_dim2, weight_dim3, weight_dim4, 1, 1, 1, weight_dim3, weight_dim2, weight_dim1, 1, 1, 1,
            output_depth, output_height, output_width
        );

        // Launch matrix rank kernel
        dim3 threadsPerBlockRank(width, height);
        dim3 numBlocksRank(batch, channels, depth);
        matrix_rank_kernel<<<numBlocksRank, threadsPerBlockRank>>>(
            d_output, d_rank, input_dim0, weight_dim0, output_depth, output_height, output_width
        );

        // Copy output and rank from device
        cudaMemcpy(output, d_output, output_dim0 * output_dim1 * output_dim2 * output_dim3 * output_dim4 * sizeof(int8_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(rank, d_rank, sizeof(int), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_weight);
        cudaFree(d_bias);
        cudaFree(d_output);
        cudaFree(d_rank);
    }
}
