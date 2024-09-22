
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void signal_shift_kernel(const float* input_tensor, float* output_tensor, int batch_size, int signal_length, int shift_amount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / signal_length;
    int signal_idx = idx % signal_length;

    if (batch_idx < batch_size && signal_idx < signal_length) {
        int shifted_idx = (signal_idx + shift_amount) % signal_length;
        output_tensor[batch_idx * signal_length + shifted_idx] = input_tensor[batch_idx * signal_length + signal_idx];
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract shift amount
    int shift_amount = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int signal_length = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * signal_length * sizeof(float));
    cudaMalloc(&d_output, batch_size * signal_length * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * signal_length * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size * signal_length + threadsPerBlock.x - 1) / threadsPerBlock.x);

    signal_shift_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, signal_length, shift_amount
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * signal_length * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
