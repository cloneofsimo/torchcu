
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

#define CHECK(status)                                              \
  {                                                              \
    if (status != cudaSuccess) {                                  \
      fprintf(stderr, "CUDA API error: %s in file %s at line %d\n", \
              cudaGetErrorString(status), __FILE__, __LINE__);     \
      exit(EXIT_FAILURE);                                        \
    }                                                             \
  }

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for convolution in frequency domain
__global__ void conv_fft_kernel(const float* input, const float* weight, const float* bias, float* output,
                                int batch_size, int in_channels, int out_channels, int in_height, int in_width,
                                int kernel_height, int kernel_width, int stride, int padding) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int out_channel = blockIdx.y * blockDim.y + threadIdx.y;
    int out_row = (batch * in_height + blockIdx.z * blockDim.z + threadIdx.z) / stride;
    int out_col = (blockIdx.z * blockDim.z + threadIdx.z) % stride;

    if (batch < batch_size && out_channel < out_channels && out_row < in_height && out_col < in_width) {
        float sum = 0.0f;
        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int row = out_row * stride + kh - padding;
                    int col = out_col * stride + kw - padding;

                    if (row >= 0 && row < in_height && col >= 0 && col < in_width) {
                        int input_idx = (batch * in_channels + in_channel) * in_height * in_width + row * in_width + col;
                        int weight_idx = (out_channel * in_channels + in_channel) * kernel_height * kernel_width + kh * kernel_width + kw;
                        sum += half_to_float(float_to_half(input[input_idx]) * float_to_half(weight[weight_idx]));
                    }
                }
            }
        }
        output[(batch * out_channels + out_channel) * in_height * in_width + out_row * in_width + out_col] = sum + bias[out_channel];
    }
}


extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);
    int input_dim2 = va_arg(args, int);
    int input_dim3 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);
    int weight_dim3 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);
    int bias_dim1 = va_arg(args, int);

    // Extract stride and padding
    int stride = va_arg(args, int);
    int padding = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_dim0;
    int in_channels = input_dim1;
    int out_channels = weight_dim0;
    int in_height = input_dim2;
    int in_width = input_dim3;
    int kernel_height = weight_dim2;
    int kernel_width = weight_dim3;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * in_channels * in_height * in_width * sizeof(float));
    cudaMalloc(&d_weight, out_channels * in_channels * kernel_height * kernel_width * sizeof(float));
    cudaMalloc(&d_bias, out_channels * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * in_height * in_width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * in_channels * in_height * in_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, out_channels * in_channels * kernel_height * kernel_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 1, 1);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (out_channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (in_height + threadsPerBlock.z - 1) / threadsPerBlock.z);

    conv_fft_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output,
        batch_size, in_channels, out_channels, in_height, in_width,
        kernel_height, kernel_width, stride, padding
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
