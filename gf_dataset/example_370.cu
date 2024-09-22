
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/transform/fft/fft.h>
#include <cutlass/transform/fft/fft_inverse.h>

// CUDA kernel for inverse FFT
__global__ void ifft_kernel_fp32(const cuComplex* input_tensor, float* output_tensor, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        cutlass::transform::fft::InversePlan<
            cutlass::layout::TensorNHWC,
            cutlass::layout::TensorNHWC,
            cutlass::complex<float>,
            float,
            cutlass::transform::fft::Cuda,
            cutlass::transform::fft::Fast,
            16,
            16,
            1
        > plan;

        plan.execute(
            input_tensor + idx,
            output_tensor + idx,
            dim,
            1
        );
    }
}

extern "C" {
    
    void torch_function(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        const cuComplex* input_tensor = va_arg(args, const cuComplex*);
        int input_tensor_dim0 = va_arg(args, int);
        int input_tensor_dim1 = va_arg(args, int);

        float* output_tensor = va_arg(args, float*);

        va_end(args);

        int batch_size = input_tensor_dim0;
        int dim = input_tensor_dim1;

        // Allocate device memory
        cuComplex *d_input;
        float *d_output;
        cudaMalloc(&d_input, batch_size * dim * sizeof(cuComplex));
        cudaMalloc(&d_output, batch_size * dim * sizeof(float));

        // Copy input data to device
        cudaMemcpy(d_input, input_tensor, batch_size * dim * sizeof(cuComplex), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 threadsPerBlock(256);
        dim3 numBlocks((batch_size * dim + threadsPerBlock.x - 1) / threadsPerBlock.x);

        ifft_kernel_fp32<<<numBlocks, threadsPerBlock>>>(
            d_input, d_output, batch_size, dim
        );

        // Copy result back to host
        cudaMemcpy(output_tensor, d_output, batch_size * dim * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
    }
}
