
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for binary cross-entropy
__global__ void bce_kernel_bf16(const float* input_tensor, const float* target_tensor, float* bce_loss,
                                int batch_size, int features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        float sum = 0.0f;
        for (int j = 0; j < features; ++j) {
            __nv_bfloat16 x = float_to_bfloat16(input_tensor[i * features + j]);
            __nv_bfloat16 y = float_to_bfloat16(target_tensor[i]);
            sum += bfloat16_to_float(-y * logf(bfloat16_to_float(expf(x))) - (1.0f - y) * logf(bfloat16_to_float(1.0f - expf(x))));
        }
        bce_loss[i] = sum / features;
    }
}

// CUDA kernel for linear transformation
__global__ void linear_kernel_bf16(const float* input_tensor, const float* weights, float* output,
                                 int batch_size, int features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        float sum = 0.0f;
        for (int j = 0; j < features; ++j) {
            __nv_bfloat16 x = float_to_bfloat16(input_tensor[i * features + j]);
            __nv_bfloat16 w = float_to_bfloat16(weights[j]);
            sum += bfloat16_to_float(__hmul(x, w));
        }
        output[i] = sum;
    }
}

// CUDA kernel for margin ranking loss
__global__ void margin_ranking_loss_kernel_bf16(const float* output1, const float* output2, float* margin_loss,
                                                int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        margin_loss[i] = fmaxf(0.0f, 1.0f - output1[i] + output2[i]);
    }
}

extern "C" {

void custom_loss_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    const float* target_tensor = va_arg(args, const float*);
    int target_tensor_dim0 = va_arg(args, int);

    const float* weights = va_arg(args, const float*);
    int weights_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int features = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_target, *d_weights, *d_bce_loss, *d_linear_output, *d_margin_loss;
    cudaMalloc(&d_input, batch_size * features * sizeof(float));
    cudaMalloc(&d_target, batch_size * sizeof(float));
    cudaMalloc(&d_weights, features * sizeof(float));
    cudaMalloc(&d_bce_loss, batch_size * sizeof(float));
    cudaMalloc(&d_linear_output, batch_size * sizeof(float));
    cudaMalloc(&d_margin_loss, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, features * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate binary cross-entropy
    bce_kernel_bf16<<<(batch_size + 255) / 256, 256>>>(d_input, d_target, d_bce_loss, batch_size, features);

    // Apply linear transformation
    linear_kernel_bf16<<<(batch_size + 255) / 256, 256>>>(d_input, d_weights, d_linear_output, batch_size, features);

    // Calculate margin ranking loss
    margin_ranking_loss_kernel_bf16<<<(batch_size + 255) / 256, 256>>>(d_linear_output, d_linear_output, d_margin_loss, batch_size);

    // Sum the losses on the device
    float *d_combined_loss;
    cudaMalloc(&d_combined_loss, sizeof(float));
    cudaMemset(d_combined_loss, 0, sizeof(float));
    cudaMemcpyAsync(d_combined_loss, d_bce_loss, batch_size * sizeof(float), cudaMemcpyDeviceToDevice, 0);
    cudaMemcpyAsync(d_combined_loss, d_margin_loss, batch_size * sizeof(float), cudaMemcpyDeviceToDevice, 0);

    // Copy result back to host
    cudaMemcpy(output, d_combined_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_weights);
    cudaFree(d_bce_loss);
    cudaFree(d_linear_output);
    cudaFree(d_margin_loss);
    cudaFree(d_combined_loss);
}

}  // extern "C"
