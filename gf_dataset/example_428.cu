
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <complex>
#include <cmath>
#include <iostream>
#include <stdarg.h>

#include "cutlass/cutlass.h"

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for IDFT and MSE calculation using bfloat16 and cutlass
__global__ void idft_mse_kernel_bf16(const float2* input, const float2* target, float* output, 
                                    int batch_size, int input_dim, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        // Allocate shared memory for input and target
        __shared__ float2 shared_input[128];
        __shared__ float2 shared_target[128];

        // Load data into shared memory
        int local_idx = threadIdx.x;
        shared_input[local_idx] = input[idx * input_dim + local_idx];
        shared_target[local_idx] = target[idx * input_dim + local_idx];

        __syncthreads();

        // Compute IDFT using cutlass
        cutlass::complex<float, cutlass::layout::RowMajor> *d_input = reinterpret_cast<cutlass::complex<float, cutlass::layout::RowMajor>*>(shared_input);
        cutlass::complex<float, cutlass::layout::RowMajor> *d_target = reinterpret_cast<cutlass::complex<float, cutlass::layout::RowMajor>*>(shared_target);
        cutlass::complex<float, cutlass::layout::RowMajor> *d_idft_input = reinterpret_cast<cutlass::complex<float, cutlass::layout::RowMajor>*>(shared_input);
        cutlass::complex<float, cutlass::layout::RowMajor> *d_idft_target = reinterpret_cast<cutlass::complex<float, cutlass::layout::RowMajor>*>(shared_target);

        // Use cutlass for IDFT
        cutlass::transform::Plan<
            cutlass::complex<float, cutlass::layout::RowMajor>, 
            cutlass::complex<float, cutlass::layout::RowMajor>,
            cutlass::transform::InverseDiscreteFourierTransform,
            cutlass::transform::KernelSize::kSize128,
            cutlass::transform::Alignment::kAlignment128
        > plan;

        plan.execute(d_idft_input, d_input, 1, input_dim);
        plan.execute(d_idft_target, d_target, 1, input_dim);

        // Calculate MSE
        float sum_squared_error = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            float real_diff = d_idft_input[i].real() - d_idft_target[i].real();
            float imag_diff = d_idft_input[i].imag() - d_idft_target[i].imag();
            sum_squared_error += real_diff * real_diff + imag_diff * imag_diff;
        }

        // Store MSE for the current batch element
        output[idx] = sum_squared_error / input_dim;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float2* input_tensor = va_arg(args, const float2*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract target tensor
    const float2* target_tensor = va_arg(args, const float2*);
    int target_tensor_dim0 = va_arg(args, int);
    int target_tensor_dim1 = va_arg(args, int);
    int target_tensor_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1 * input_tensor_dim2; 
    int output_dim = 1; // Since we're calculating MSE

    // Allocate device memory
    float2 *d_input, *d_target;
    float *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float2));
    cudaMalloc(&d_target, batch_size * input_dim * sizeof(float2));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, batch_size * input_dim * sizeof(float2), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(128);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    idft_mse_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_target, d_output, batch_size, input_dim, output_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_output);
}

}  // extern "C"
