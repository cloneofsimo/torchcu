
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for calculating contrastive loss
__global__ void contrastive_loss_kernel(const float* anchor, const float* positive, const float* negative, float* loss, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        float distance_ap = sqrtf((anchor[i] - positive[i]) * (anchor[i] - positive[i]));
        float distance_an = sqrtf((anchor[i] - negative[i]) * (anchor[i] - negative[i]));
        loss[i] = fmaxf(0.0f, distance_ap - distance_an + 1.0f);  // Margin of 1.0
    }
}

extern "C" {

void contrastive_loss_example(int num_args, ...) {
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

    // Extract output tensor (assuming it's preallocated)
    float* loss = va_arg(args, float*);

    va_end(args);

    int batch_size = anchor_dim0; // Assuming all tensors have the same batch size

    // Allocate device memory
    float *d_anchor, *d_positive, *d_negative, *d_loss;
    cudaMalloc(&d_anchor, batch_size * sizeof(float));
    cudaMalloc(&d_positive, batch_size * sizeof(float));
    cudaMalloc(&d_negative, batch_size * sizeof(float));
    cudaMalloc(&d_loss, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_anchor, anchor, batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positive, positive, batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative, negative, batch_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    contrastive_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_anchor, d_positive, d_negative, d_loss, batch_size
    );

    // Copy result back to host
    cudaMemcpy(loss, d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_anchor);
    cudaFree(d_positive);
    cudaFree(d_negative);
    cudaFree(d_loss);
}

}  // extern "C"
