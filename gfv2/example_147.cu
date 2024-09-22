
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

extern "C" {

__global__ void diagflat_multiply_kernel(const float* input, const float* weight, float* output,
                                         int batch_size, int channels, int width, int diag_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels * width) {
        int b = idx / (channels * width);
        int c = (idx % (channels * width)) / width;
        int w = idx % width;

        float sum = 0.0f;
        for (int i = 0; i < diag_size; ++i) {
            sum += input[(b * channels + c) * width + (w + i) % width] * weight[i];
        }
        output[idx] = sum;
    }
}

void conv_fft_diagflat_inplace(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);
    int input_dim2 = va_arg(args, int);
    int input_dim3 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);

    va_end(args);

    // Allocate device memory
    float* d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, input_dim0 * input_dim1 * input_dim2 * input_dim3 * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * sizeof(float));
    cudaMalloc(&d_output, input_dim0 * input_dim1 * input_dim2 * input_dim3 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_dim0 * input_dim1 * input_dim2 * input_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution using FFT on device
    cufftHandle plan;
    cufftPlanMany(&plan, 1, &input_dim3, &input_dim3, 1, input_dim3 * sizeof(float), 0, 0,
                  1, &input_dim3, &input_dim3, 1, input_dim3 * sizeof(float), 0, 0,
                  CUFFT_C2C, input_dim0 * input_dim1 * input_dim2);

    // Allocate complex data on device
    cufftComplex* d_input_complex;
    cudaMalloc(&d_input_complex, input_dim0 * input_dim1 * input_dim2 * input_dim3 * sizeof(cufftComplex));

    // Copy real data to complex data (assuming real input)
    cudaMemcpy(d_input_complex, d_input, input_dim0 * input_dim1 * input_dim2 * input_dim3 * sizeof(float), cudaMemcpyDeviceToDevice);

    // Perform forward FFT
    cufftExecC2C(plan, (cufftComplex*)d_input_complex, (cufftComplex*)d_input_complex, CUFFT_FORWARD);

    // Perform diagonal matrix multiplication
    diagflat_multiply_kernel<<<(input_dim0 * input_dim1 * input_dim2 * input_dim3 + 128 - 1) / 128, 128>>>(
        d_input_complex, d_weight, d_output, input_dim0, input_dim1, input_dim3, weight_dim0);

    // Perform inverse FFT
    cufftExecC2C(plan, (cufftComplex*)d_output, (cufftComplex*)d_output, CUFFT_INVERSE);

    // Copy result back to host
    cudaMemcpy(d_input, d_output, input_dim0 * input_dim1 * input_dim2 * input_dim3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cufftDestroy(plan);
    cudaFree(d_input_complex);
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

} // extern "C"
