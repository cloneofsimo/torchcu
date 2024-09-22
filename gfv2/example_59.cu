
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <complex>
#include <stdarg.h>

// Helper function for ifftshift
template <typename T>
__device__ __forceinline__ T ifftshift_element(T element, int n, int i) {
    if (i < n / 2) {
        return element;
    } else {
        return element - (i - n / 2 + 1);
    }
}

// CUDA kernel for ifftshift
template <typename T>
__global__ void ifftshift_kernel(T* data, int n, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        int idx = i * n + j;
        data[idx] = ifftshift_element(data[idx], n, j);
    }
}

// CUDA kernel for complex FFT
template <typename T>
__global__ void complex_fft_kernel(T* data, int n, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i * n * k + j * k;

    if (i < m && j < n) {
        // Apply FFT to each 2D slice
        cufftHandle plan;
        cufftPlan2d(&plan, k, k, CUFFT_C2C);

        cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(data + idx),
                     reinterpret_cast<cufftComplex*>(data + idx), CUFFT_FORWARD);

        cufftDestroy(plan);
    }
}

extern "C" {

void complex_shift_and_fft(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const std::complex<float>* input_tensor = va_arg(args, const std::complex<float>*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    std::complex<float>* output_tensor = va_arg(args, std::complex<float>*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int height = input_tensor_dim1;
    int width = input_tensor_dim2;

    // Allocate device memory
    std::complex<float>* d_input, *d_output;
    cudaMalloc(&d_input, batch_size * height * width * sizeof(std::complex<float>));
    cudaMalloc(&d_output, batch_size * height * width * sizeof(std::complex<float>));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * height * width * sizeof(std::complex<float>), cudaMemcpyHostToDevice);

    // Launch ifftshift kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    ifftshift_kernel<<<numBlocks, threadsPerBlock>>>(d_input, width, height, batch_size);

    // Launch complex FFT kernel
    numBlocks = ((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    complex_fft_kernel<<<numBlocks, threadsPerBlock>>>(d_input, width, height, batch_size);

    // Copy result back to host
    cudaMemcpy(output_tensor, d_input, batch_size * height * width * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

} // extern "C"
