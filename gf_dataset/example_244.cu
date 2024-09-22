
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cutlass.h"

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for batch matrix multiplication (using cutlass)
__global__ void bmm_kernel(const float* input, const float* target, float* output, int batch_size, int num_classes, int num_features) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        for (int i = 0; i < num_classes; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < num_features; ++j) {
                sum += input[batch_idx * num_classes * num_features + i * num_features + j] *
                       target[batch_idx * num_classes + j];
            }
            output[batch_idx * num_classes + i] = sum;
        }
    }
}

// CUDA kernel for sigmoid focal loss calculation (using cutlass)
__global__ void focal_loss_kernel(const half* output, const half* target, const half* weights, float* loss, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            half p = __expf(output[batch_idx * num_classes + i]);
            half loss_val = weights[batch_idx * num_classes + i] * (target[batch_idx * num_classes + i] * -__logf(p) + (1 - target[batch_idx * num_classes + i]) * -__logf(1 - p));
            sum += half_to_float((1 - p) * (1 - p) * loss_val);
        }
        loss[batch_idx] = sum;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    const float* target_tensor = va_arg(args, const float*);
    int target_tensor_dim0 = va_arg(args, int);
    int target_tensor_dim1 = va_arg(args, int);

    const float* weights_tensor = va_arg(args, const float*);
    int weights_tensor_dim0 = va_arg(args, int);
    int weights_tensor_dim1 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int num_classes = input_tensor_dim1;
    int num_features = input_tensor_dim2;

    // Allocate device memory
    float *d_input, *d_target, *d_weights, *d_output;
    half *d_output_fp16;
    cudaMalloc(&d_input, batch_size * num_classes * num_features * sizeof(float));
    cudaMalloc(&d_target, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_weights, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_output, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_output_fp16, batch_size * num_classes * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * num_classes * num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights_tensor, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    // Launch bmm kernel
    dim3 threadsPerBlock(32, 1);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);

    bmm_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_target, d_output, batch_size, num_classes, num_features);

    // Convert output to half
    cudaMemcpy(d_output_fp16, d_output, batch_size * num_classes * sizeof(half), cudaMemcpyDeviceToDevice);

    // Launch focal loss kernel
    focal_loss_kernel<<<numBlocks, threadsPerBlock>>>(d_output_fp16, reinterpret_cast<const half*>(d_target), reinterpret_cast<const half*>(d_weights), d_output, batch_size, num_classes);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaFree(d_output_fp16);
}

}  // extern "C"
