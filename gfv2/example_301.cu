
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

extern "C" {

__global__ void mean_pad_slice_fp16_kernel(const half* input_tensor, half pad_value, int slice_start, int slice_end,
                                        int input_size, half* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) {
        // Apply padding
        int padded_idx = idx + slice_start;
        if (padded_idx < 0 || padded_idx >= slice_end) {
            output[idx] = pad_value;
        } else {
            output[idx] = input_tensor[padded_idx];
        }
    }
}

__global__ void mean_kernel(const half* input_tensor, int input_size, half* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) {
        output[0] += input_tensor[idx];
    }
}

void mean_pad_slice_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor_fp32 = va_arg(args, const float*);
    int input_size = va_arg(args, int);
    int input_dim = va_arg(args, int);

    // Extract padding value
    float pad_value_fp32 = va_arg(args, float);
    half pad_value = __float2half_rn(pad_value_fp32);

    // Extract slice indices
    int slice_start = va_arg(args, int);
    int slice_end = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output_fp32 = va_arg(args, float*);
    half* output = reinterpret_cast<half*>(output_fp32);

    va_end(args);

    // Allocate device memory
    half *d_input, *d_output_pad, *d_output_mean;
    cudaMalloc(&d_input, input_size * sizeof(half));
    cudaMalloc(&d_output_pad, input_size * sizeof(half));
    cudaMalloc(&d_output_mean, sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor_fp32, input_size * sizeof(half), cudaMemcpyHostToDevice);

    // Launch padding kernel
    int threadsPerBlock = 256;
    int numBlocks = (input_size + threadsPerBlock - 1) / threadsPerBlock;
    mean_pad_slice_fp16_kernel<<<numBlocks, threadsPerBlock>>>(d_input, pad_value, slice_start, slice_end, input_size, d_output_pad);

    // Launch mean kernel
    mean_kernel<<<1, 256>>>(d_output_pad, input_size, d_output_mean);

    // Copy result back to host
    cudaMemcpy(output, d_output_mean, sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output_pad);
    cudaFree(d_output_mean);
}
}
