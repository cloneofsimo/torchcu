
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// CUDA kernel for calculating NLL loss, rounding, and converting to fp16
__global__ void nll_loss_round_fp16_kernel(const float* input_tensor, const int* target, 
                                             half* output, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float loss = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            if (i == target[idx]) {
                loss -= logf(input_tensor[idx * num_classes + i]);
            }
        }
        output[idx] = __int2half_rn(roundf(loss));  // Round and convert to fp16
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

    // Extract target tensor
    const int* target = va_arg(args, const int*);
    int target_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    half* output = va_arg(args, half*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int num_classes = input_tensor_dim1;

    // Allocate device memory
    float *d_input;
    int *d_target;
    half *d_output;
    cudaMalloc(&d_input, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_target, batch_size * sizeof(int));
    cudaMalloc(&d_output, batch_size * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    nll_loss_round_fp16_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_target, d_output, batch_size, num_classes
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_output);
}

}  // extern "C"
