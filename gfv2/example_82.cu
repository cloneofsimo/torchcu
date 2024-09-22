
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <complex>
#include <math.h>
#include <stdarg.h> 

#define PI 3.14159265358979323846

// Helper functions for complex numbers
__device__ __forceinline__ std::complex<float> complex_mul(std::complex<float> a, std::complex<float> b) {
    return std::complex<float>(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

// Function to perform FFT on a 1D array (real-to-complex)
__device__ void fft1D(std::complex<float> *x, int n) {
    if (n == 1) return;

    std::complex<float> even[n / 2], odd[n / 2];
    for (int i = 0; i < n / 2; ++i) {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    fft1D(even, n / 2);
    fft1D(odd, n / 2);

    for (int i = 0; i < n / 2; ++i) {
        float angle = -2 * PI * i / n;
        std::complex<float> factor(cos(angle), sin(angle));
        x[i] = even[i] + complex_mul(factor, odd[i]);
        x[i + n / 2] = even[i] - complex_mul(factor, odd[i]);
    }
}

// Function to perform inverse FFT on a 1D array (complex-to-real)
__device__ void ifft1D(std::complex<float> *x, int n) {
    // Conjugate the input array
    for (int i = 0; i < n; ++i) {
        x[i] = std::conj(x[i]);
    }

    // Perform FFT
    fft1D(x, n);

    // Conjugate again and divide by N
    for (int i = 0; i < n; ++i) {
        x[i] = std::conj(x[i]) / n;
    }
}

// CUDA kernel for int8 FFT convolution 1D
__global__ void int8_fft_conv1d_kernel(const int8_t *input, const int8_t *weight, const float *bias, float *output, 
                                         int batch_size, int in_channels, int out_channels, int input_size, 
                                         int kernel_size, int stride, int padding, int dilation, int groups, int mode) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_channel_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch_size && out_channel_idx < out_channels) {
        int out_idx = batch_idx * out_channels + out_channel_idx;

        // Padding
        int padding_start = (mode == 0) ? padding : 0; // Same padding
        int padding_end = (mode == 0) ? padding : padding * 2; // Same padding

        int input_size_padded = input_size + padding_start + padding_end;

        // Output size calculation
        int output_size = (input_size_padded - dilation * (kernel_size - 1) - 1) / stride + 1;

        // Allocate memory for FFT on device
        std::complex<float> *d_input_fft, *d_weight_fft, *d_output_fft;
        cudaMalloc(&d_input_fft, input_size_padded * sizeof(std::complex<float>));
        cudaMalloc(&d_weight_fft, kernel_size * sizeof(std::complex<float>));
        cudaMalloc(&d_output_fft, output_size * sizeof(std::complex<float>));

        // Copy input to device
        cudaMemcpy(d_input_fft, input + batch_idx * in_channels * input_size, input_size_padded * sizeof(float), cudaMemcpyHostToDevice);

        // Zero-pad input for FFT
        for (int i = input_size; i < input_size_padded; ++i) {
            ((float *)d_input_fft)[i] = 0.0f;
        }

        // Copy weight to device
        cudaMemcpy(d_weight_fft, weight + out_channel_idx * kernel_size, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

        // Perform FFT
        fft1D(d_input_fft, input_size_padded);
        fft1D(d_weight_fft, kernel_size);

        // Multiply in frequency domain
        for (int i = 0; i < output_size; ++i) {
            for (int k = 0; k < kernel_size; ++k) {
                int input_idx = i * stride + k * dilation; 

                if (input_idx >= 0 && input_idx < input_size_padded) {
                    d_output_fft[i] = complex_mul(d_output_fft[i], d_weight_fft[k]);
                    d_output_fft[i] = complex_mul(d_output_fft[i], d_input_fft[input_idx]);
                }
            }
        }

        // Inverse FFT
        ifft1D(d_output_fft, output_size);

        // Copy output from device
        cudaMemcpy(output + out_idx, d_output_fft, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        // Apply bias
        output[out_idx] += bias[out_channel_idx];

        // Free device memory
        cudaFree(d_input_fft);
        cudaFree(d_weight_fft);
        cudaFree(d_output_fft);
    }
}

extern "C" {

void int8_fft_conv1d(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const int8_t* input = va_arg(args, const int8_t*);
    int batch_size = va_arg(args, int);
    int in_channels = va_arg(args, int);
    int input_size = va_arg(args, int);

    // Extract weight tensor
    const int8_t* weight = va_arg(args, const int8_t*);
    int out_channels = va_arg(args, int);
    int kernel_size = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);

    // Extract other arguments
    int stride = va_arg(args, int);
    int padding = va_arg(args, int);
    int dilation = va_arg(args, int);
    int groups = va_arg(args, int);
    int mode = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    // Calculate output size
    int output_size = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (out_channels + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int8_fft_conv1d_kernel<<<numBlocks, threadsPerBlock>>>(
        input, weight, bias, output, batch_size, in_channels, out_channels, input_size,
        kernel_size, stride, padding, dilation, groups, mode
    );
}

} // extern "C"
