
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for calculating triplet margin loss
__global__ void triplet_loss_kernel(const float* anchor, const float* positive, const float* negative, 
                                        float* output, int batch_size, int embedding_dim, float margin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < batch_size) {
        float anchor_positive_distance = 0.0f;
        float anchor_negative_distance = 0.0f;

        for (int j = 0; j < embedding_dim; ++j) {
            float anchor_val = anchor[i * embedding_dim + j];
            float positive_val = positive[i * embedding_dim + j];
            float negative_val = negative[i * embedding_dim + j];

            anchor_positive_distance += (anchor_val - positive_val) * (anchor_val - positive_val);
            anchor_negative_distance += (anchor_val - negative_val) * (anchor_val - negative_val);
        }

        anchor_positive_distance = sqrtf(anchor_positive_distance);
        anchor_negative_distance = sqrtf(anchor_negative_distance);

        float loss = fmaxf(0.0f, anchor_positive_distance - anchor_negative_distance + margin);
        output[i] = loss;
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
    float margin = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = anchor_dim0;
    int embedding_dim = anchor_dim1;

    // Allocate device memory
    float *d_anchor, *d_positive, *d_negative, *d_output;
    cudaMalloc(&d_anchor, batch_size * embedding_dim * sizeof(float));
    cudaMalloc(&d_positive, batch_size * embedding_dim * sizeof(float));
    cudaMalloc(&d_negative, batch_size * embedding_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_anchor, anchor, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positive, positive, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative, negative, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    triplet_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_anchor, d_positive, d_negative, d_output, batch_size, embedding_dim, margin
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
