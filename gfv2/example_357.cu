
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

// CUDA kernel for multi-margin loss with constant padding
__global__ void multi_margin_loss_kernel(const float* input_tensor, const float* weight, float* output,
                                        int target, int padding, float pad_value, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Handle padding
    if (idx >= padding && idx < input_size + padding) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            float val = input_tensor[idx + i];
            float w = weight[i];
            sum += fmaxf(0.0f, w * (val - input_tensor[idx + target] + 1.0f));
        }
        output[0] = sum;
    } else if (idx < padding || idx >= input_size + padding) {
        output[0] = pad_value;
    }
}

extern "C" {

void multi_margin_loss_with_padding(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);

    // Extract target
    int target = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    // Extract padding and pad_value
    int padding = va_arg(args, int);
    float pad_value = va_arg(args, float);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((input_tensor_dim0 + 2 * padding + threadsPerBlock.x - 1) / threadsPerBlock.x);

    multi_margin_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, target, padding, pad_value, input_tensor_dim0
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

} // extern "C"
