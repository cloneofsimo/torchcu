
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for 2D convolution
__global__ void conv2d_kernel(const float* input, const float* weight, const float* bias, float* output,
                              int batch_size, int in_channels, int out_channels,
                              int input_height, int input_width,
                              int kernel_height, int kernel_width,
                              int padding) {

    int b = blockIdx.z; // Batch index
    int o = blockIdx.y; // Output channel index
    int h = blockIdx.x * blockDim.x + threadIdx.x; // Height index
    int w = threadIdx.y; // Width index

    if (h < input_height && w < input_width) {
        float sum = bias[o];
        for (int i = 0; i < in_channels; ++i) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int ih = h + kh - padding;
                    int iw = w + kw - padding;
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        sum += weight[o * in_channels * kernel_height * kernel_width + i * kernel_height * kernel_width + kh * kernel_width + kw] *
                               input[b * in_channels * input_height * input_width + i * input_height * input_width + ih * input_width + iw];
                    }
                }
            }
        }
        output[b * out_channels * input_height * input_width + o * input_height * input_width + h * input_width + w] = sum;
    }
}

// CUDA kernel for MSE loss calculation
__global__ void mse_loss_kernel(const float* output, const float* gt, float* loss,
                               int batch_size, int out_channels,
                               int input_height, int input_width) {

    int b = blockIdx.z; // Batch index
    int o = blockIdx.y; // Output channel index
    int h = blockIdx.x * blockDim.x + threadIdx.x; // Height index
    int w = threadIdx.y; // Width index

    if (h < input_height && w < input_width) {
        float diff = output[b * out_channels * input_height * input_width + o * input_height * input_width + h * input_width + w] -
                    gt[b * out_channels * input_height * input_width + o * input_height * input_width + h * input_width + w];
        atomicAdd(loss, diff * diff);
    }
}

// CUDA kernel for backpropagation of MSE loss
__global__ void mse_loss_backward_kernel(const float* output, const float* gt, float* input_grad,
                                       int batch_size, int out_channels,
                                       int input_height, int input_width) {

    int b = blockIdx.z; // Batch index
    int o = blockIdx.y; // Output channel index
    int h = blockIdx.x * blockDim.x + threadIdx.x; // Height index
    int w = threadIdx.y; // Width index

    if (h < input_height && w < input_width) {
        float diff = output[b * out_channels * input_height * input_width + o * input_height * input_width + h * input_width + w] -
                    gt[b * out_channels * input_height * input_width + o * input_height * input_width + h * input_width + w];
        atomicAdd(&input_grad[b * out_channels * input_height * input_width + o * input_height * input_width + h * input_width + w], 2 * diff);
    }
}

extern "C" {

void conv2d_with_gt_backward_return_one(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract inputs
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);
    int input_dim2 = va_arg(args, int);
    int input_dim3 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);
    int weight_dim3 = va_arg(args, int);

    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    const float* gt = va_arg(args, const float*);
    int gt_dim0 = va_arg(args, int);
    int gt_dim1 = va_arg(args, int);
    int gt_dim2 = va_arg(args, int);
    int gt_dim3 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_dim0;
    int in_channels = input_dim1;
    int out_channels = weight_dim0;
    int input_height = input_dim2;
    int input_width = input_dim3;
    int kernel_height = weight_dim2;
    int kernel_width = weight_dim3;
    int padding = 1; // Assuming padding = 1

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_gt, *d_output;
    cudaMalloc(&d_input, batch_size * in_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_weight, out_channels * in_channels * kernel_height * kernel_width * sizeof(float));
    cudaMalloc(&d_bias, out_channels * sizeof(float));
    cudaMalloc(&d_gt, batch_size * out_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * input_height * input_width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * in_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, out_channels * in_channels * kernel_height * kernel_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gt, gt, batch_size * out_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch convolution kernel
    dim3 block_conv(16, 16);
    dim3 grid_conv((input_height + block_conv.x - 1) / block_conv.x, (out_channels + block_conv.y - 1) / block_conv.y, batch_size);
    conv2d_kernel<<<grid_conv, block_conv>>>(d_input, d_weight, d_bias, d_output,
                                             batch_size, in_channels, out_channels,
                                             input_height, input_width,
                                             kernel_height, kernel_width,
                                             padding);

    // Allocate device memory for loss and input gradient
    float *d_loss, *d_input_grad;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMalloc(&d_input_grad, batch_size * in_channels * input_height * input_width * sizeof(float));

    // Launch MSE loss kernel
    dim3 block_loss(16, 16);
    dim3 grid_loss((input_height + block_loss.x - 1) / block_loss.x, (out_channels + block_loss.y - 1) / block_loss.y, batch_size);
    mse_loss_kernel<<<grid_loss, block_loss>>>(d_output, d_gt, d_loss,
                                             batch_size, out_channels,
                                             input_height, input_width);

    // Launch MSE loss backward kernel
    mse_loss_backward_kernel<<<grid_loss, block_loss>>>(d_output, d_gt, d_input_grad,
                                                     batch_size, out_channels,
                                                     input_height, input_width);

    // Copy input gradient back to host
    cudaMemcpy(output, d_input_grad, batch_size * in_channels * input_height * input_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_gt);
    cudaFree(d_output);
    cudaFree(d_loss);
    cudaFree(d_input_grad);
}

}  // extern "C"
