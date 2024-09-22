
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <complex>
#include <math.h>
#include <stdarg.h> 

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for complex 2D convolution using FFT in FP16
__global__ void complex_conv2d_fft_fp16_kernel(const float2* input_tensor, const float2* weight, const float* bias, 
                                              float* output, int batch_size, int in_channels, int out_channels, 
                                              int input_height, int input_width, int kernel_height, int kernel_width) {

    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int out_channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int output_height_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_channel_idx < out_channels && output_height_idx < input_height) {
        float2 sum_real = make_float2(0.0f, 0.0f);
        float2 sum_imag = make_float2(0.0f, 0.0f);

        for (int in_channel_idx = 0; in_channel_idx < in_channels; ++in_channel_idx) {
            for (int kernel_height_idx = 0; kernel_height_idx < kernel_height; ++kernel_height_idx) {
                for (int kernel_width_idx = 0; kernel_width_idx < kernel_width; ++kernel_width_idx) {
                    int input_height_idx = output_height_idx + kernel_height_idx - (kernel_height / 2);
                    int input_width_idx = threadIdx.x + kernel_width_idx - (kernel_width / 2);

                    if (input_height_idx >= 0 && input_height_idx < input_height && input_width_idx >= 0 && input_width_idx < input_width) {
                        float2 input_val = make_float2(half_to_float(input_tensor[(batch_idx * in_channels + in_channel_idx) * input_height * input_width + input_height_idx * input_width + input_width_idx].x), 
                                                        half_to_float(input_tensor[(batch_idx * in_channels + in_channel_idx) * input_height * input_width + input_height_idx * input_width + input_width_idx].y));
                        float2 weight_val = make_float2(half_to_float(weight[(out_channel_idx * in_channels + in_channel_idx) * kernel_height * kernel_width + kernel_height_idx * kernel_width + kernel_width_idx].x), 
                                                        half_to_float(weight[(out_channel_idx * in_channels + in_channel_idx) * kernel_height * kernel_width + kernel_height_idx * kernel_width + kernel_width_idx].y));

                        sum_real.x += input_val.x * weight_val.x - input_val.y * weight_val.y;
                        sum_real.y += input_val.x * weight_val.y + input_val.y * weight_val.x;
                    }
                }
            }
        }

        // Apply bias and softplus activation
        float bias_val = half_to_float(bias[out_channel_idx]);
        sum_real.x += bias_val;
        sum_real.y += 0.0f;

        float output_val = expf(sum_real.x);
        output_val = output_val / (1.0f + output_val);

        output[(batch_idx * out_channels + out_channel_idx) * input_height * input_width + output_height_idx * input_width + threadIdx.x] = output_val;
    }
}

extern "C" {

void complex_conv2d_fft_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float2* input_tensor = va_arg(args, const float2*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract weight tensors
    const float2* weight = va_arg(args, const float2*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);
    int weight_dim3 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int out_channels = weight_dim0;
    int input_height = input_tensor_dim2;
    int input_width = input_tensor_dim3;
    int kernel_height = weight_dim2;
    int kernel_width = weight_dim3;

    // Launch kernel
    dim3 threadsPerBlock(input_width, 1, 1);
    dim3 numBlocks(input_height, out_channels, batch_size);

    complex_conv2d_fft_fp16_kernel<<<numBlocks, threadsPerBlock>>>(
        input_tensor, weight, bias, output,
        batch_size, in_channels, out_channels,
        input_height, input_width, kernel_height, kernel_width
    );

    cudaDeviceSynchronize();
}

}
