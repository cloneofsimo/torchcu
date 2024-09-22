
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

// Helper functions for FP16 conversion
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for cross-entropy calculation
__global__ void cross_entropy_kernel(const half* input, const char* target, const half* weights, float* loss,
                                     int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int label = target[idx];
        half prob = input[idx * num_classes + label];
        loss[0] += half_to_float(-logf(half_to_float(prob)) * weights[label]);
    }
}

extern "C" {

void my_complex_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract target tensor
    const char* target = va_arg(args, const char*);
    int target_dim = va_arg(args, int);

    // Extract weights tensor
    const float* weights = va_arg(args, const float*);
    int weights_dim = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int num_classes = input_tensor_dim1;

    // Allocate device memory for FP16 data
    half *d_input, *d_weights;
    cudaMalloc(&d_input, batch_size * num_classes * sizeof(half));
    cudaMalloc(&d_weights, num_classes * sizeof(half));

    // Allocate device memory for output
    float *d_loss;
    cudaMalloc(&d_loss, sizeof(float));

    // Copy input and weights to device in FP16
    cudaMemcpy(d_input, input_tensor, batch_size * num_classes * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, num_classes * sizeof(half), cudaMemcpyHostToDevice);

    // Launch cross-entropy kernel
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);
    cross_entropy_kernel<<<grid, block>>>(d_input, target, d_weights, d_loss, batch_size, num_classes);

    // Scale the loss
    cudaMemcpy(d_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    *d_loss *= 2;
    cudaMemcpy(d_loss, d_loss, sizeof(float), cudaMemcpyHostToDevice);

    // Divide the loss by the sum of weights
    float sum_weights = 0;
    for (int i = 0; i < num_classes; i++) {
        sum_weights += weights[i];
    }
    *d_loss /= sum_weights;

    // Copy result back to host
    cudaMemcpy(output, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_loss);
}

}  // extern "C"
