
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for scatter-add operation
__global__ void scatter_add_kernel(const float* input, const int* indices, float* output, 
                                    int batch_size, int in_channels, int height, int width) {
    int b = blockIdx.z * blockDim.z + threadIdx.z;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && c < in_channels && h < height) {
        int index = indices[b * in_channels * height * width + c * height * width + h];
        output[index * in_channels * height * width + c * height * width + h] += input[b * in_channels * height * width + c * height * width + h];
    }
}

// CUDA kernel for separable 2D convolution (depthwise and pointwise)
__global__ void conv2d_kernel(const float* input, const float* weight_depth, const float* weight_point, 
                              float* output, int batch_size, int in_channels, int out_channels, int height, int width,
                              int kernel_size, int padding) {
    int b = blockIdx.z * blockDim.z + threadIdx.z;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && c < out_channels && h < height) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            for (int l = 0; l < kernel_size; l++) {
                int input_h = h + k - padding;
                int input_w = c + l - padding;

                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    sum += input[b * in_channels * height * width + c * height * width + input_h * width + input_w] * weight_depth[c * kernel_size * kernel_size + k * kernel_size + l];
                }
            }
        }

        // Pointwise convolution
        for (int i = 0; i < in_channels; i++) {
            sum += output[b * out_channels * height * width + i * height * width + h * width + c] * weight_point[c * in_channels + i];
        }

        output[b * out_channels * height * width + c * height * width + h * width + c] = sum;
    }
}

// CUDA kernel for conv2d transpose (backpropagation through separable convolution)
__global__ void conv2d_transpose_kernel(const float* input, const float* weight_depth, const float* weight_point, 
                                          float* output, int batch_size, int in_channels, int out_channels, int height, int width,
                                          int kernel_size, int padding) {
    int b = blockIdx.z * blockDim.z + threadIdx.z;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && c < in_channels && h < height) {
        float sum = 0.0f;

        // Pointwise convolution transpose
        for (int i = 0; i < out_channels; i++) {
            sum += input[b * out_channels * height * width + i * height * width + h * width + c] * weight_point[i * in_channels + c];
        }

        // Depthwise convolution transpose
        for (int k = 0; k < kernel_size; k++) {
            for (int l = 0; l < kernel_size; l++) {
                int input_h = h + k - padding;
                int input_w = c + l - padding;

                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    sum += output[b * in_channels * height * width + c * height * width + input_h * width + input_w] * weight_depth[c * kernel_size * kernel_size + k * kernel_size + l];
                }
            }
        }

        output[b * in_channels * height * width + c * height * width + h * width + c] = sum;
    }
}

// CUDA kernel for gathering elements along the batch dimension
__global__ void gather_kernel(const float* input, const int* indices, float* output,
                               int batch_size, int in_channels, int height, int width) {
    int b = blockIdx.z * blockDim.z + threadIdx.z;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && c < in_channels && h < height) {
        int index = indices[b * in_channels * height * width + c * height * width + h];
        output[b * in_channels * height * width + c * height * width + h] = input[index * in_channels * height * width + c * height * width + h];
    }
}

extern "C" {

void scatter_add_conv2d_backprop(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract weights tensor
    const float* weight_depth = va_arg(args, const float*);
    int weight_depth_dim0 = va_arg(args, int);
    int weight_depth_dim1 = va_arg(args, int);
    int weight_depth_dim2 = va_arg(args, int);
    int weight_depth_dim3 = va_arg(args, int);

    const float* weight_point = va_arg(args, const float*);
    int weight_point_dim0 = va_arg(args, int);
    int weight_point_dim1 = va_arg(args, int);

    // Extract indices tensor
    const int* indices = va_arg(args, const int*);
    int indices_dim0 = va_arg(args, int);
    int indices_dim1 = va_arg(args, int);
    int indices_dim2 = va_arg(args, int);
    int indices_dim3 = va_arg(args, int);

    // Extract output shape
    int output_shape_dim0 = va_arg(args, int);
    int output_shape_dim1 = va_arg(args, int);
    int output_shape_dim2 = va_arg(args, int);
    int output_shape_dim3 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Get dimensions
    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int height = input_tensor_dim2;
    int width = input_tensor_dim3;
    int out_channels = weight_point_dim0;

    // Allocate device memory
    float *d_input, *d_weight_depth, *d_weight_point, *d_output;
    int *d_indices;
    cudaMalloc(&d_input, batch_size * in_channels * height * width * sizeof(float));
    cudaMalloc(&d_weight_depth, out_channels * weight_depth_dim1 * weight_depth_dim2 * weight_depth_dim3 * sizeof(float));
    cudaMalloc(&d_weight_point, out_channels * in_channels * sizeof(float));
    cudaMalloc(&d_output, batch_size * in_channels * height * width * sizeof(float));
    cudaMalloc(&d_indices, batch_size * in_channels * height * width * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, batch_size * in_channels * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_depth, weight_depth, out_channels * weight_depth_dim1 * weight_depth_dim2 * weight_depth_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_point, weight_point, out_channels * in_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices, batch_size * in_channels * height * width * sizeof(int), cudaMemcpyHostToDevice);

    // Scatter-add operation
    dim3 scatter_add_threadsPerBlock(16, 16, 1);
    dim3 scatter_add_numBlocks((height + scatter_add_threadsPerBlock.x - 1) / scatter_add_threadsPerBlock.x,
                               (in_channels + scatter_add_threadsPerBlock.y - 1) / scatter_add_threadsPerBlock.y,
                               (batch_size + scatter_add_threadsPerBlock.z - 1) / scatter_add_threadsPerBlock.z);

    scatter_add_kernel<<<scatter_add_numBlocks, scatter_add_threadsPerBlock>>>(
        d_input, d_indices, d_output, batch_size, in_channels, height, width
    );

    // Separable convolution
    dim3 conv2d_threadsPerBlock(16, 16, 1);
    dim3 conv2d_numBlocks((height + conv2d_threadsPerBlock.x - 1) / conv2d_threadsPerBlock.x,
                           (out_channels + conv2d_threadsPerBlock.y - 1) / conv2d_threadsPerBlock.y,
                           (batch_size + conv2d_threadsPerBlock.z - 1) / conv2d_threadsPerBlock.z);

    conv2d_kernel<<<conv2d_numBlocks, conv2d_threadsPerBlock>>>(
        d_output, d_weight_depth, d_weight_point, d_output, batch_size, in_channels, out_channels, height, width, 
        weight_depth_dim2, 1
    );

    // Backpropagation through convolution
    conv2d_transpose_kernel<<<conv2d_numBlocks, conv2d_threadsPerBlock>>>(
        d_output, d_weight_depth, d_weight_point, d_output, batch_size, out_channels, in_channels, height, width,
        weight_depth_dim2, 1
    );

    conv2d_transpose_kernel<<<conv2d_numBlocks, conv2d_threadsPerBlock>>>(
        d_output, d_weight_depth, d_weight_point, d_output, batch_size, in_channels, out_channels, height, width,
        weight_depth_dim2, 1
    );

    // Backpropagation through scatter-add (gather operation)
    dim3 gather_threadsPerBlock(16, 16, 1);
    dim3 gather_numBlocks((height + gather_threadsPerBlock.x - 1) / gather_threadsPerBlock.x,
                           (in_channels + gather_threadsPerBlock.y - 1) / gather_threadsPerBlock.y,
                           (batch_size + gather_threadsPerBlock.z - 1) / gather_threadsPerBlock.z);

    gather_kernel<<<gather_numBlocks, gather_threadsPerBlock>>>(
        d_output, d_indices, d_output, batch_size, in_channels, height, width
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * in_channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight_depth);
    cudaFree(d_weight_point);
    cudaFree(d_output);
    cudaFree(d_indices);
}

}  // extern "C"
