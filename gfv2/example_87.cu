
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

__global__ void grouped_conv2d_kernel(const float* input, const float* weight, const float* bias, float* output, int batch, int in_channels, int out_channels, int groups, int height, int width, int kernel_size) {
    int b = blockIdx.z * blockDim.z + threadIdx.z;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    int oy = blockIdx.x * blockDim.x + threadIdx.x;
    int ox = threadIdx.x;

    if (b < batch && g < groups && oy < height && ox < width) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int iy = oy + ky - kernel_size / 2;
                int ix = ox + kx - kernel_size / 2;
                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    int i_idx = (b * in_channels + g * (in_channels / groups) + iy * width + ix);
                    int w_idx = (g * (out_channels / groups) + (ky * kernel_size + kx) * (in_channels / groups));
                    sum += input[i_idx] * weight[w_idx];
                }
            }
        }
        output[(b * out_channels + g * (out_channels / groups) + oy * width + ox)] = sum + bias[g * (out_channels / groups)];
    }
}

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

__global__ void avg_pool2d_kernel(const float* input, float* output, int batch, int channels, int in_height, int in_width, int out_height, int out_width) {
    int b = blockIdx.z * blockDim.z + threadIdx.z;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int oy = blockIdx.x * blockDim.x + threadIdx.x;
    int ox = threadIdx.x;

    if (b < batch && c < channels && oy < out_height && ox < out_width) {
        float sum = 0.0f;
        for (int ky = 0; ky < 2; ++ky) {
            for (int kx = 0; kx < 2; ++kx) {
                int iy = oy * 2 + ky;
                int ix = ox * 2 + kx;
                if (iy >= 0 && iy < in_height && ix >= 0 && ix < in_width) {
                    sum += input[(b * channels + c) * in_height * in_width + iy * in_width + ix];
                }
            }
        }
        output[(b * channels + c) * out_height * out_width + oy * out_width + ox] = sum / 4.0f;
    }
}

__global__ void spectral_bandwidth_kernel(const float* input, float* output, int batch, int channels, int height, int width) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = threadIdx.y;

    if (b < batch && c < channels) {
        float sum = 0.0f;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = (b * channels + c) * height * width + y * width + x;
                sum += fabsf(input[idx]);
            }
        }
        output[b * channels + c] = sum / (height * width);
    }
}

__global__ void rms_energy_kernel(const float* input, float* output, int batch, int channels, int height, int width) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = threadIdx.y;

    if (b < batch && c < channels) {
        float sum = 0.0f;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = (b * channels + c) * height * width + y * width + x;
                sum += input[idx] * input[idx];
            }
        }
        output[b * channels + c] = sqrtf(sum / (height * width));
    }
}


extern "C" {

void grouped_conv_and_analysis(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input = va_arg(args, const float*);
    int input_shape0 = va_arg(args, int);
    int input_shape1 = va_arg(args, int);
    int input_shape2 = va_arg(args, int);
    int input_shape3 = va_arg(args, int);
    const float* weight = va_arg(args, const float*);
    int weight_shape0 = va_arg(args, int);
    int weight_shape1 = va_arg(args, int);
    int weight_shape2 = va_arg(args, int);
    int weight_shape3 = va_arg(args, int);
    const float* bias = va_arg(args, const float*);
    int bias_shape0 = va_arg(args, int);
    float* output = va_arg(args, float*);
    float* spectral_bandwidth = va_arg(args, float*);
    float* rms_energy = va_arg(args, float*);

    va_end(args);

    // Convolution
    int batch = input_shape0;
    int in_channels = input_shape1;
    int out_channels = weight_shape0;
    int groups = 4;
    int kernel_size = weight_shape2;
    int height = input_shape2;
    int width = input_shape3;

    dim3 threadsPerBlock(width, 1, 1);
    dim3 numBlocks(height, groups, batch);
    grouped_conv2d_kernel<<<numBlocks, threadsPerBlock>>>(input, weight, bias, output, batch, in_channels, out_channels, groups, height, width, kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    int output_size = batch * out_channels * (height - kernel_size + 1) * (width - kernel_size + 1);
    relu_kernel<<<(output_size + 255) / 256, 256>>>(output, output_size);
    cudaDeviceSynchronize();

    // Average Pooling
    int out_height = (height - kernel_size + 1) / 2;
    int out_width = (width - kernel_size + 1) / 2;
    avg_pool2d_kernel<<<numBlocks, threadsPerBlock>>>(output, output, batch, out_channels, height - kernel_size + 1, width - kernel_size + 1, out_height, out_width);
    cudaDeviceSynchronize();

    // Spectral Bandwidth
    spectral_bandwidth_kernel<<<(batch + 255) / 256, 256>>>(output, spectral_bandwidth, batch, out_channels, out_height, out_width);
    cudaDeviceSynchronize();

    // RMS Energy
    rms_energy_kernel<<<(batch + 255) / 256, 256>>>(output, rms_energy, batch, out_channels, out_height, out_width);
    cudaDeviceSynchronize();
}

}
