
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <pywt.h>

#define THREADS_PER_BLOCK 256

// DWT kernel
__global__ void dwt_kernel(const float* input, float* cA, float* cH, float* cV, float* cD, int n, const pywt::Wavelet& wavelet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pywt::dwt_coeffs(input + i, wavelet, cA + i, cH + i, cV + i, cD + i);
    }
}

// IDWT kernel
__global__ void idwt_kernel(const float* cA, const float* cH, const float* cV, const float* cD, float* output, int n, const pywt::Wavelet& wavelet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pywt::idwt_coeffs(cA + i, cH + i, cV + i, cD + i, wavelet, output + i);
    }
}

// Gradient kernel
__global__ void gradient_kernel(const float* cA_grad, const float* cH_grad, const float* cV_grad, const float* cD_grad, float* grad_x, int n, const pywt::Wavelet& wavelet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pywt::idwt_coeffs(cA_grad + i, cH_grad + i, cV_grad + i, cD_grad + i, wavelet, grad_x + i);
    }
}

extern "C" {
    void wavelet_transform_resynthesis(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract input
        const float* input = va_arg(args, const float*);
        int n = va_arg(args, int);
        int m = va_arg(args, int);

        // Extract output
        float* output = va_arg(args, float*);

        va_end(args);

        // Allocate device memory
        float *d_input, *d_cA, *d_cH, *d_cV, *d_cD, *d_grad_x;
        cudaMalloc(&d_input, n * m * sizeof(float));
        cudaMalloc(&d_cA, n / 2 * sizeof(float));
        cudaMalloc(&d_cH, n / 2 * sizeof(float));
        cudaMalloc(&d_cV, n / 2 * sizeof(float));
        cudaMalloc(&d_cD, n / 2 * sizeof(float));
        cudaMalloc(&d_grad_x, n * m * sizeof(float));

        // Copy data to device
        cudaMemcpy(d_input, input, n * m * sizeof(float), cudaMemcpyHostToDevice);

        // Set wavelet
        pywt::Wavelet wavelet("db4");

        // Launch DWT kernel
        dwt_kernel<<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_input, d_cA, d_cH, d_cV, d_cD, n, wavelet);

        // Launch IDWT kernel
        idwt_kernel<<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_cA, d_cH, d_cV, d_cD, d_grad_x, n, wavelet);

        // Copy output back to host
        cudaMemcpy(output, d_grad_x, n * m * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_cA);
        cudaFree(d_cH);
        cudaFree(d_cV);
        cudaFree(d_cD);
        cudaFree(d_grad_x);
    }
}
