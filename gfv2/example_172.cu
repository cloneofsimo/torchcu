
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

__global__ void normalize_kernel(const float* input, half* output, int batch_size, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * embedding_dim) {
        int batch_idx = idx / embedding_dim;
        int feature_idx = idx % embedding_dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < embedding_dim; ++i) {
            sum_sq += input[batch_idx * embedding_dim + i] * input[batch_idx * embedding_dim + i];
        }
        output[idx] = float_to_half(input[idx] / sqrtf(sum_sq));
    }
}

__global__ void simclr_loss_kernel(const half* z1, const half* z2, float* loss, int batch_size, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float max_similarity = -1e9f;
        float positive_similarity = -1e9f;
        for (int i = 0; i < batch_size; ++i) {
            float similarity = 0.0f;
            for (int j = 0; j < embedding_dim; ++j) {
                similarity += half_to_float(z1[idx * embedding_dim + j]) * half_to_float(z2[i * embedding_dim + j]);
            }
            if (i == idx) {
                positive_similarity = similarity;
            } else if (similarity > max_similarity) {
                max_similarity = similarity;
            }
        }
        loss[idx] = -logf(expf(positive_similarity) / (expf(positive_similarity) + expf(max_similarity)));
    }
}

extern "C" {

void simclr_loss_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* z1 = va_arg(args, const float*);
    int z1_dim0 = va_arg(args, int);
    int z1_dim1 = va_arg(args, int);

    const float* z2 = va_arg(args, const float*);
    int z2_dim0 = va_arg(args, int);
    int z2_dim1 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = z1_dim0;
    int embedding_dim = z1_dim1;

    // Allocate device memory
    half* d_z1, *d_z2;
    float* d_loss;
    cudaMalloc(&d_z1, batch_size * embedding_dim * sizeof(half));
    cudaMalloc(&d_z2, batch_size * embedding_dim * sizeof(half));
    cudaMalloc(&d_loss, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_z1, z1, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z2, z2, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Normalize embeddings on the device
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size * embedding_dim + threadsPerBlock.x - 1) / threadsPerBlock.x);
    normalize_kernel<<<numBlocks, threadsPerBlock>>>(d_z1, d_z1, batch_size, embedding_dim);
    normalize_kernel<<<numBlocks, threadsPerBlock>>>(d_z2, d_z2, batch_size, embedding_dim);

    // Calculate SimCLR loss on the device
    numBlocks = (batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    simclr_loss_kernel<<<numBlocks, threadsPerBlock>>>(d_z1, d_z2, d_loss, batch_size, embedding_dim);

    // Copy result back to host
    cudaMemcpy(output, d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_z1);
    cudaFree(d_z2);
    cudaFree(d_loss);
}

}  // extern "C"
