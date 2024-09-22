
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

__global__ void signal_processing_kernel_fp16(const half* input_tensor, const int window_size, half* output_tensor, int signal_length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < signal_length) {
        half sum = 0.0h;
        for (int j = 0; j < window_size; ++j) {
            int index = i + j - window_size;
            if (index >= 0 && index < signal_length) {
                sum += input_tensor[index] * (1.0f / window_size); // Use fp16 directly here
            }
        }
        output_tensor[i] = __expf(-__fmaf_rn(sum, -1.0f, 0.0f));
    }
}

extern "C" {

void signal_processing_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int window_size = va_arg(args, int);

    float* output_tensor = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    half *d_input, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * sizeof(half));
    cudaMalloc(&d_output, input_tensor_dim0 * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(128);
    dim3 numBlocks((input_tensor_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x);

    signal_processing_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_input, window_size, d_output, input_tensor_dim0
    );

    // Copy result back to host
    cudaMemcpy(output_tensor, d_output, input_tensor_dim0 * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"