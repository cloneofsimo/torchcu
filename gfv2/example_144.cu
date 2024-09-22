
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for 2D convolution
__global__ void conv2d_kernel_bf16(const float* input, const float* weight, const float* bias, float* output,
                                  int batch_size, int in_channels, int out_channels, int kernel_size,
                                  int input_height, int input_width, int padding) {

    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && out_y < input_height && out_x < input_width) {
        float sum = 0.0f;
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int ky = -padding; ky < kernel_size - padding; ++ky) {
                    for (int kx = -padding; kx < kernel_size - padding; ++kx) {
                        int in_y = out_y + ky;
                        int in_x = out_x + kx;

                        if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                            __nv_bfloat16 a = float_to_bfloat16(input[batch * in_channels * input_height * input_width + ic * input_height * input_width + in_y * input_width + in_x]);
                            __nv_bfloat16 b = float_to_bfloat16(weight[oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + ky * kernel_size + kx]);
                            sum += bfloat16_to_float(__hmul(a, b));
                        }
                    }
                }
            }
            sum += bfloat16_to_float(bias[oc]);
        }
        output[batch * out_channels * input_height * input_width + oc * input_height * input_width + out_y * input_width + out_x] = 1.0f / (1.0f + expf(-sum));
    }
}

__global__ void sigmoid_focal_loss_kernel(const float* output, const float* target, float* loss,
                                         int batch_size, int out_channels, int height, int width) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && out_y < height && out_x < width) {
        int idx = batch * out_channels * height * width + out_y * width + out_x;
        float p = output[idx];
        float t = target[idx];
        loss[idx] = (1.0f - p) * (1.0f - p) * -t * logf(p) - (1.0f - t) * logf(1.0f - p);
    }
}

extern "C" {

void conv2d_sigmoid_focal_loss(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);
    int weight_dim3 = va_arg(args, int);

    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    const float* target = va_arg(args, const float*);
    int target_dim0 = va_arg(args, int);
    int target_dim1 = va_arg(args, int);
    int target_dim2 = va_arg(args, int);
    int target_dim3 = va_arg(args, int);

    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int out_channels = weight_dim0;
    int kernel_size = weight_dim2;
    int input_height = input_tensor_dim2;
    int input_width = input_tensor_dim3;

    int padding = (kernel_size - 1) / 2;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output, *d_target;
    cudaMalloc(&d_input, batch_size * in_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_weight, out_channels * in_channels * kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_bias, out_channels * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_target, batch_size * out_channels * input_height * input_width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * in_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, out_channels * in_channels * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, batch_size * out_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((input_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    conv2d_kernel_bf16<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_bias, d_output,
                                                 batch_size, in_channels, out_channels, kernel_size,
                                                 input_height, input_width, padding);

    sigmoid_focal_loss_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_target, d_output,
                                                            batch_size, out_channels, input_height, input_width);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * out_channels * input_height * input_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaFree(d_target);
}

}  // extern "C"
