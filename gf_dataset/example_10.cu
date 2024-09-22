
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h> 

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for triplet margin loss calculation
__global__ void triplet_margin_loss_kernel(const float* anchor, const float* positive, const float* negative, float* output, int batch_size, float margin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        float dist_ap = 0.0f;
        float dist_an = 0.0f;
        for (int j = 0; j < 3; ++j) { // Assuming each feature has 3 dimensions
            dist_ap += (anchor[i * 3 + j] - positive[i * 3 + j]) * (anchor[i * 3 + j] - positive[i * 3 + j]);
            dist_an += (anchor[i * 3 + j] - negative[i * 3 + j]) * (anchor[i * 3 + j] - negative[i * 3 + j]);
        }

        dist_ap = sqrtf(dist_ap);
        dist_an = sqrtf(dist_an);
        
        // Calculate loss
        output[i] = fmaxf(0.0f, dist_ap - dist_an + margin);
    }
}

extern "C" {

void triplet_loss_net(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* anchor = va_arg(args, const float*);
    int anchor_dim0 = va_arg(args, int);
    int anchor_dim1 = va_arg(args, int);

    const float* positive = va_arg(args, const float*);
    int positive_dim0 = va_arg(args, int);
    int positive_dim1 = va_arg(args, int);

    const float* negative = va_arg(args, const float*);
    int negative_dim0 = va_arg(args, int);
    int negative_dim1 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    // Extract margin
    float margin = va_arg(args, double);

    va_end(args);

    int batch_size = anchor_dim0;

    // Allocate device memory
    float *d_anchor, *d_positive, *d_negative, *d_output;
    cudaMalloc(&d_anchor, batch_size * 3 * sizeof(float));
    cudaMalloc(&d_positive, batch_size * 3 * sizeof(float));
    cudaMalloc(&d_negative, batch_size * 3 * sizeof(float));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_anchor, anchor, batch_size * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positive, positive, batch_size * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative, negative, batch_size * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    triplet_margin_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_anchor, d_positive, d_negative, d_output, batch_size, margin
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_anchor);
    cudaFree(d_positive);
    cudaFree(d_negative);
    cudaFree(d_output);
}

}  // extern "C"
