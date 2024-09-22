
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <complex>
#include <stdarg.h>

// Helper function for complex multiplication
__device__ __forceinline__ std::complex<float> complex_mul(const std::complex<float>& a, const std::complex<float>& b) {
    return std::complex<float>(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

// CUDA kernel for 3D convolution using FFT
__global__ void conv3d_fft_kernel(const int8_t* input_tensor, const int8_t* kernel, float* output,
                                 int batch_size, int in_channels, int in_depth, int in_height, int in_width,
                                 int kernel_depth, int kernel_height, int kernel_width) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_depth = blockIdx.y * blockDim.y + threadIdx.y;
    int out_height = blockIdx.z * blockDim.z + threadIdx.z;
    int out_width = threadIdx.w;

    if (batch_idx < batch_size && out_depth < in_depth - kernel_depth + 1 &&
        out_height < in_height - kernel_height + 1 && out_width < in_width - kernel_width + 1) {

        // Calculate indices for input and kernel
        int input_start_depth = out_depth;
        int input_start_height = out_height;
        int input_start_width = out_width;

        // Initialize sum to 0
        float sum = 0.0f;

        // Perform convolution using FFT
        for (int kd = 0; kd < kernel_depth; ++kd) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int input_idx = ((batch_idx * in_channels + in_channels - 1) * in_depth + input_start_depth + kd) * in_height + input_start_height + kh) * in_width + input_start_width + kw;
                    int kernel_idx = (kd * kernel_height + kh) * kernel_width + kw;
                    sum += (float) input_tensor[input_idx] * (float) kernel[kernel_idx];
                }
            }
        }

        output[((batch_idx * in_channels + in_channels - 1) * (in_depth - kernel_depth + 1) + out_depth) * (in_height - kernel_height + 1) + out_height) * (in_width - kernel_width + 1) + out_width] = sum;
    }
}

extern "C" {

void conv3d_fft_einsum_squeeze_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const int8_t* input_tensor = va_arg(args, const int8_t*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);
    int input_tensor_dim4 = va_arg(args, int);

    // Extract kernel tensor
    const int8_t* kernel = va_arg(args, const int8_t*);
    int kernel_dim0 = va_arg(args, int);
    int kernel_dim1 = va_arg(args, int);
    int kernel_dim2 = va_arg(args, int);
    int kernel_dim3 = va_arg(args, int);
    int kernel_dim4 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int in_depth = input_tensor_dim2;
    int in_height = input_tensor_dim3;
    int in_width = input_tensor_dim4;
    int kernel_depth = kernel_dim2;
    int kernel_height = kernel_dim3;
    int kernel_width = kernel_dim4;

    // Allocate device memory
    int8_t *d_input, *d_kernel;
    float *d_output;
    cudaMalloc(&d_input, batch_size * in_channels * in_depth * in_height * in_width * sizeof(int8_t));
    cudaMalloc(&d_kernel, kernel_dim0 * kernel_dim1 * kernel_depth * kernel_height * kernel_width * sizeof(int8_t));
    cudaMalloc(&d_output, batch_size * in_channels * (in_depth - kernel_depth + 1) * (in_height - kernel_height + 1) * (in_width - kernel_width + 1) * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * in_channels * in_depth * in_height * in_width * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_dim0 * kernel_dim1 * kernel_depth * kernel_height * kernel_width * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (in_depth - kernel_depth + 1 + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (in_height - kernel_height + 1 + threadsPerBlock.z - 1) / threadsPerBlock.z);

    conv3d_fft_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_kernel, d_output, batch_size, in_channels, in_depth, in_height, in_width, kernel_depth, kernel_height, kernel_width
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * in_channels * (in_depth - kernel_depth + 1) * (in_height - kernel_height + 1) * (in_width - kernel_width + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

}  // extern "C"
