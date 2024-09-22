
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

__global__ void rms_energy_kernel(const float* input_tensor, const float* weight, float* output,
                                 int batch_size, int input_dim1, int input_dim2, int weight_dim1, int weight_dim2) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim1; ++i) {
            for (int j = 0; j < input_dim2; ++j) {
                float val = input_tensor[batch_idx * input_dim1 * input_dim2 + i * input_dim2 + j];
                float w = weight[batch_idx * weight_dim1 * weight_dim2 + i * weight_dim2 + j];
                sum += val * val * w * w;
            }
        }
        output[batch_idx] = sqrtf(sum / (input_dim1 * input_dim2));
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
    int input_tensor_dim2 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * weight_dim2 * sizeof(float));
    cudaMalloc(&d_output, input_tensor_dim0 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * weight_dim2 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((input_tensor_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x);

    rms_energy_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, input_tensor_dim0, input_tensor_dim1, input_tensor_dim2, weight_dim1, weight_dim2
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor_dim0 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
