
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

extern "C" {

__global__ void hinge_embedding_loss_quantized_kernel(const float* input_tensor, const float* target_tensor, float* output, int batch_size, int embedding_dim, float margin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        float sum = 0.0f;
        for (int j = 0; j < embedding_dim; ++j) {
            sum += roundf(input_tensor[i * embedding_dim + j]) * target_tensor[i];
        }
        output[i] = fmaxf(margin + sum - 1.0f, 0.0f);
    }
}

void hinge_embedding_loss_quantized(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    const float* target_tensor = va_arg(args, const float*);
    int target_tensor_dim0 = va_arg(args, int);
    int target_tensor_dim1 = va_arg(args, int);

    float margin = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int embedding_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_target, *d_output;
    cudaMalloc(&d_input, batch_size * embedding_dim * sizeof(float));
    cudaMalloc(&d_target, batch_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, batch_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    hinge_embedding_loss_quantized_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_target, d_output, batch_size, embedding_dim, margin
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_output);
}

} // extern "C"
