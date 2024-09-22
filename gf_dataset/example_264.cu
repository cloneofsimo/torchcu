
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <complex>
#include <cufft.h>
#include <iostream>
#include <stdarg.h> 

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for 3D reflection padding
__global__ void reflection_padding_kernel(const float* input, float* padded_input,
                                         int batch_size, int in_channels, int D, int H, int W,
                                         int kernel_D, int kernel_H, int kernel_W) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < batch_size && c < in_channels && d < D) {
        int pad_D = kernel_D / 2;
        int pad_H = kernel_H / 2;
        int pad_W = kernel_W / 2;

        int d_index = d + pad_D;
        int h_index = threadIdx.z + pad_H;
        int w_index = threadIdx.w + pad_W;

        if (d_index >= 0 && d_index < D + 2 * pad_D &&
            h_index >= 0 && h_index < H + 2 * pad_H &&
            w_index >= 0 && w_index < W + 2 * pad_W) {
            // Reflection padding
            int d_offset = (d_index >= D) ? 2 * (D - 1) - d_index : d_index;
            int h_offset = (h_index >= H) ? 2 * (H - 1) - h_index : h_index;
            int w_offset = (w_index >= W) ? 2 * (W - 1) - w_index : w_index;

            padded_input[((b * in_channels + c) * (D + 2 * pad_D) + d_offset) * (H + 2 * pad_H) + h_offset] = 
                input[(b * in_channels + c) * D * H + d_offset * H + h_offset];
        }
    }
}

// CUDA kernel for 3D convolution using FFT
__global__ void conv3d_fft_kernel_bf16(const __nv_bfloat16* padded_input, const __nv_bfloat16* weight, 
                                       __nv_bfloat16* output, 
                                       int batch_size, int in_channels, int out_channels,
                                       int D, int H, int W, 
                                       int kernel_D, int kernel_H, int kernel_W, float scale,
                                       const __nv_bfloat16* bias) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < batch_size && c < out_channels && d < D) {
        __nv_bfloat16 sum = float_to_bfloat16(0.0f);
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_D; ++kd) {
                for (int kh = 0; kh < kernel_H; ++kh) {
                    for (int kw = 0; kw < kernel_W; ++kw) {
                        int input_index = ((b * in_channels + ic) * (D + kernel_D - 1) + d + kd) * (H + kernel_H - 1) +
                                          threadIdx.z + kh;
                        int weight_index = (c * in_channels + ic) * kernel_D * kernel_H * kernel_W + 
                                            kd * kernel_H * kernel_W + kh * kernel_W + kw;

                        sum += __hmul(padded_input[input_index], weight[weight_index]);
                    }
                }
            }
        }
        sum = bfloat16_to_float(sum) * scale;
        sum += bfloat16_to_float(bias[c]);
        output[((b * out_channels + c) * D + d) * H + threadIdx.z] = float_to_bfloat16(fmaxf(sum, 0.0f));
    }
}

// CUDA kernel for 3D FFT
__global__ void fft_kernel(const float* data, cufftComplex* complex_data, 
                             int batch_size, int in_channels, int D, int H, int W) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < batch_size && c < in_channels && d < D) {
        complex_data[(b * in_channels + c) * D * H * W + d * H * W + threadIdx.z * W + threadIdx.w] = 
            make_cufftComplex(data[(b * in_channels + c) * D * H + d * H + threadIdx.z * W + threadIdx.w], 0.0f);
    }
}

// CUDA kernel for 3D inverse FFT
__global__ void ifft_kernel(const cufftComplex* complex_data, float* data, 
                             int batch_size, int in_channels, int D, int H, int W) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < batch_size && c < in_channels && d < D) {
        data[(b * in_channels + c) * D * H + d * H + threadIdx.z * W + threadIdx.w] = 
            complex_data[(b * in_channels + c) * D * H * W + d * H * W + threadIdx.z * W + threadIdx.w].x;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
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

    // Extract scale
    float scale = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int D = input_tensor_dim2;
    int H = input_tensor_dim3;
    int W = input_tensor_dim4;

    int out_channels = weight_dim0;
    int kernel_D = weight_dim2;
    int kernel_H = weight_dim3;
    int kernel_W = weight_dim4;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    __nv_bfloat16 *d_padded_input, *d_output_bf16;

    cudaMalloc(&d_input, batch_size * in_channels * D * H * W * sizeof(float));
    cudaMalloc(&d_weight, out_channels * in_channels * kernel_D * kernel_H * kernel_W * sizeof(float));
    cudaMalloc(&d_bias, out_channels * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * D * H * W * sizeof(float));
    cudaMalloc(&d_padded_input, batch_size * in_channels * (D + kernel_D - 1) * (H + kernel_H - 1) * (W + kernel_W - 1) * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output_bf16, batch_size * out_channels * D * H * W * sizeof(__nv_bfloat16));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * in_channels * D * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, out_channels * in_channels * kernel_D * kernel_H * kernel_W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Perform reflection padding
    dim3 padding_threads(W, H, D);
    dim3 padding_blocks(batch_size * in_channels, 1, 1);
    reflection_padding_kernel<<<padding_blocks, padding_threads>>>(d_input, reinterpret_cast<float*>(d_padded_input),
                                                         batch_size, in_channels, D, H, W,
                                                         kernel_D, kernel_H, kernel_W);

    // 3D FFT on padded input and weight
    cufftHandle plan_input, plan_weight;
    cufftPlanMany(&plan_input, CUFFT_R2C, 1, &W, &H, &D, 1, W, &H, &D,
                   CUFFT_C2R, 1, &W, &H, &D, 1, W, &H, &D, CUFFT_REAL, 
                   CUFFT_COMPLEX, batch_size * in_channels, CUFFT_COMPLEX,
                   batch_size * in_channels, sizeof(float), sizeof(cufftComplex),
                   sizeof(cufftComplex));
    cufftPlanMany(&plan_weight, CUFFT_R2C, 1, &kernel_W, &kernel_H, &kernel_D, 1, kernel_W, &kernel_H, &kernel_D,
                   CUFFT_C2R, 1, &kernel_W, &kernel_H, &kernel_D, 1, kernel_W, &kernel_H, &kernel_D, CUFFT_REAL, 
                   CUFFT_COMPLEX, out_channels * in_channels, CUFFT_COMPLEX,
                   out_channels * in_channels, sizeof(float), sizeof(cufftComplex),
                   sizeof(cufftComplex));
    cufftComplex *d_complex_input, *d_complex_weight;
    cudaMalloc(&d_complex_input, batch_size * in_channels * (D + kernel_D - 1) * (H + kernel_H - 1) * (W + kernel_W - 1) * sizeof(cufftComplex));
    cudaMalloc(&d_complex_weight, out_channels * in_channels * kernel_D * kernel_H * kernel_W * sizeof(cufftComplex));
    dim3 fft_threads(W, H, D);
    dim3 fft_blocks(batch_size * in_channels, 1, 1);
    fft_kernel<<<fft_blocks, fft_threads>>>(reinterpret_cast<const float*>(d_padded_input), d_complex_input,
                                      batch_size, in_channels, D + kernel_D - 1, H + kernel_H - 1, W + kernel_W - 1);
    fft_kernel<<<fft_blocks, fft_threads>>>(reinterpret_cast<const float*>(d_weight), d_complex_weight,
                                      out_channels * in_channels, 1, kernel_D, kernel_H, kernel_W);
    cufftExecR2C(plan_input, reinterpret_cast<float*>(d_padded_input), d_complex_input);
    cufftExecR2C(plan_weight, reinterpret_cast<float*>(d_weight), d_complex_weight);

    // Element-wise multiplication in frequency domain
    dim3 mul_threads(W, H, D);
    dim3 mul_blocks(batch_size * out_channels, 1, 1);
    conv3d_fft_kernel_bf16<<<mul_blocks, mul_threads>>>(reinterpret_cast<__nv_bfloat16*>(d_complex_input), reinterpret_cast<__nv_bfloat16*>(d_complex_weight),
                                                        reinterpret_cast<__nv_bfloat16*>(d_complex_input),
                                                        batch_size, in_channels, out_channels,
                                                        D + kernel_D - 1, H + kernel_H - 1, W + kernel_W - 1,
                                                        kernel_D, kernel_H, kernel_W, scale,
                                                        reinterpret_cast<__nv_bfloat16*>(d_bias));

    // Inverse FFT
    cufftExecC2R(plan_input, d_complex_input, reinterpret_cast<float*>(d_output_bf16));
    cufftDestroy(plan_input);
    cufftDestroy(plan_weight);
    cudaFree(d_complex_input);
    cudaFree(d_complex_weight);

    // Copy result back to host
    cudaMemcpy(output, reinterpret_cast<float*>(d_output_bf16), batch_size * out_channels * D * H * W * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaFree(d_padded_input);
    cudaFree(d_output_bf16);
}

}  // extern "C"
