
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/gemm.h>

#include <iostream>

using namespace cutlass;

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for cosine similarity calculation using int8
__global__ void cosine_similarity_kernel(const int8_t* anchor, const int8_t* positive, const int8_t* negative, 
                                       float* similarity_matrix, int batch_size, int feature_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < batch_size + 1) {
        float dot_product = 0.0f;
        float anchor_norm = 0.0f;
        float other_norm = 0.0f;

        for (int i = 0; i < feature_dim; ++i) {
            dot_product += anchor[row * feature_dim + i] * (col == 0 ? positive[row * feature_dim + i] : negative[(row * batch_size + (col - 1)) * feature_dim + i]);
            anchor_norm += anchor[row * feature_dim + i] * anchor[row * feature_dim + i];
            other_norm += (col == 0 ? positive[row * feature_dim + i] : negative[(row * batch_size + (col - 1)) * feature_dim + i]) * (col == 0 ? positive[row * feature_dim + i] : negative[(row * batch_size + (col - 1)) * feature_dim + i]);
        }

        similarity_matrix[row * (batch_size + 1) + col] = dot_product / (sqrtf(anchor_norm) * sqrtf(other_norm));
    }
}

// CUDA kernel for cross-entropy calculation
__global__ void cross_entropy_kernel(const float* positive_similarity, const float* negative_similarity, 
                                     float* loss, int batch_size, float temperature) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float exp_pos = expf(positive_similarity[idx] / temperature);
        float exp_neg = expf(negative_similarity[idx] / temperature);
        loss[idx] = -logf(exp_pos / (exp_pos + exp_neg));
    }
}

extern "C" {

void supervised_contrastive_loss_int8_kernel(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract tensors
    const int8_t* anchor = va_arg(args, const int8_t*);
    const int8_t* positive = va_arg(args, const int8_t*);
    const int8_t* negative = va_arg(args, const int8_t*);

    // Extract dimensions
    int batch_size = va_arg(args, int);
    int feature_dim = va_arg(args, int);

    // Extract output tensor
    float* loss = va_arg(args, float*);

    // Extract temperature
    float temperature = va_arg(args, float);

    va_end(args);

    // Allocate device memory
    int size = (batch_size + 1) * batch_size;
    float *d_similarity_matrix, *d_positive_similarity, *d_negative_similarity;
    cudaMalloc(&d_similarity_matrix, size * sizeof(float));
    cudaMalloc(&d_positive_similarity, batch_size * sizeof(float));
    cudaMalloc(&d_negative_similarity, batch_size * sizeof(float));

    // Launch cosine similarity kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cosine_similarity_kernel<<<numBlocks, threadsPerBlock>>>(
        anchor, positive, negative, d_similarity_matrix, batch_size, feature_dim
    );

    // Extract positive and negative similarity from the matrix
    cudaMemcpy(d_positive_similarity, d_similarity_matrix, batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_negative_similarity, d_similarity_matrix + batch_size, batch_size * batch_size * sizeof(float), cudaMemcpyDeviceToDevice);

    // Launch cross-entropy kernel
    numBlocks = (batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    cross_entropy_kernel<<<numBlocks, threadsPerBlock>>>(
        d_positive_similarity, d_negative_similarity, loss, batch_size, temperature
    );

    // Free device memory
    cudaFree(d_similarity_matrix);
    cudaFree(d_positive_similarity);
    cudaFree(d_negative_similarity);
}

}  // extern "C"
