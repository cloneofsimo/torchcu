
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for computing the triplet loss
__global__ void triplet_loss_kernel(const float* anchor, const float* positive, const float* negative, float* loss, 
                                        int batch_size, float margin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        float dist_ap = 0.0f;
        float dist_an = 0.0f;

        for (int i = 0; i < 128; i++) {
            float diff_ap = anchor[idx * 128 + i] - positive[idx * 128 + i];
            float diff_an = anchor[idx * 128 + i] - negative[idx * 128 + i];
            dist_ap += diff_ap * diff_ap;
            dist_an += diff_an * diff_an;
        }

        dist_ap = sqrtf(dist_ap);
        dist_an = sqrtf(dist_an);
        
        loss[idx] = fmaxf(dist_ap - dist_an + margin, 0.0f);
    }
}

extern "C" {

void triplet_loss_function(int num_args, ...) {
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

    // Extract margin
    float margin = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* loss = va_arg(args, float*);

    va_end(args);

    int batch_size = anchor_dim0;

    // Allocate device memory
    float *d_anchor, *d_positive, *d_negative, *d_loss;
    cudaMalloc(&d_anchor, batch_size * 128 * sizeof(float));
    cudaMalloc(&d_positive, batch_size * 128 * sizeof(float));
    cudaMalloc(&d_negative, batch_size * 128 * sizeof(float));
    cudaMalloc(&d_loss, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_anchor, anchor, batch_size * 128 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positive, positive, batch_size * 128 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative, negative, batch_size * 128 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256); 
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    triplet_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_anchor, d_positive, d_negative, d_loss, batch_size, margin
    );

    // Copy result back to host
    cudaMemcpy(loss, d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_anchor);
    cudaFree(d_positive);
    cudaFree(d_negative);
    cudaFree(d_loss);

    // Compute the mean of the loss
    float sum = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        sum += loss[i];
    }
    loss[0] = sum / batch_size; 
}

}  // extern "C"

