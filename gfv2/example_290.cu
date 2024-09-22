
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

extern "C" {

__global__ void triplet_loss_int8_kernel(const int8_t* anchor, const int8_t* positive, const int8_t* negative,
                                        float margin, float* output, int batch_size, int embedding_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        float distance_ap = 0.0f;
        float distance_an = 0.0f;
        for (int j = 0; j < embedding_dim; j++) {
            int8_t a = anchor[i * embedding_dim + j];
            int8_t p = positive[i * embedding_dim + j];
            int8_t n = negative[i * embedding_dim + j];
            distance_ap += (a - p) * (a - p);
            distance_an += (a - n) * (a - n);
        }
        float loss = fmaxf(distance_ap - distance_an + margin, 0.0f);
        output[i] = loss;
    }
}

void triplet_loss_int8_function(int num_args, ...) {
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
    int8_t* d_anchor, *d_positive, *d_negative;
    float* d_output;
    cudaMalloc(&d_anchor, batch_size * embedding_dim * sizeof(int8_t));
    cudaMalloc(&d_positive, batch_size * embedding_dim * sizeof(int8_t));
    cudaMalloc(&d_negative, batch_size * embedding_dim * sizeof(int8_t));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_anchor, anchor, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positive, positive, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative, negative, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(128);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    triplet_loss_int8_kernel<<<numBlocks, threadsPerBlock>>>(
        d_anchor, d_positive, d_negative, margin, d_output, batch_size, embedding_dim);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_anchor);
    cudaFree(d_positive);
    cudaFree(d_negative);
    cudaFree(d_output);

    // Calculate mean loss
    float mean_loss = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        mean_loss += output[i];
    }
    mean_loss /= batch_size;
    output[0] = mean_loss;
}

}  // extern "C"
