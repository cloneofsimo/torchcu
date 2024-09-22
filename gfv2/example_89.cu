
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h> 

#define PI 3.14159265358979323846

// Function to compute the complex exponent
__device__ __forceinline__ complex<float> cexp(complex<float> z) {
    return complex<float>(cos(z.imag()), sin(z.imag())) * expf(z.real());
}

// CUDA kernel for real-to-complex FFT using radix-2 Cooley-Tukey algorithm
__global__ void rfft_kernel(const float* input, complex<float>* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Handle only the first half of the input (due to symmetry)
    if (i < n / 2) {
        complex<float> sum = 0.0f;
        for (int k = 0; k < n / 2; k++) {
            float angle = -2.0f * PI * k * i / n;
            sum += input[2 * k] * cexp(complex<float>(0.0f, angle));
        }
        output[i] = sum;
    }
}

extern "C" {

void rfft_int8_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    char* output = va_arg(args, char*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input;
    complex<float> *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * (input_dim / 2) * sizeof(complex<float>));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size * input_dim / 2 + threadsPerBlock.x - 1) / threadsPerBlock.x);

    rfft_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, input_dim);

    // Copy result back to host (real part only)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < input_dim / 2; j++) {
            output[i * input_dim / 2 + j] = (char) round(d_output[i * input_dim / 2 + j].real());
        }
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}
