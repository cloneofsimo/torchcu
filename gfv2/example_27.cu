
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_operation.h"
#include "cutlass/epilogue/threadblock/epilogue_threadblock.h"
#include "cutlass/epilogue/threadblock/linear_combine.h"

#include "cutlass/util/tensor_view.h"

using namespace cutlass;

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}


// CUDA kernel for triplet loss with attention (using cutlass for efficient matrix multiplication)
template <typename T>
__global__ void triplet_loss_with_attention_kernel(const T* anchor, const T* positive, const T* negative, 
                                                    const T* attention_weights, T* output, int batch_size, int embedding_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        // Apply attention weights
        T weighted_anchor[embedding_dim];
        T weighted_positive[embedding_dim];
        T weighted_negative[embedding_dim];

        for (int j = 0; j < embedding_dim; ++j) {
            weighted_anchor[j] = anchor[i * embedding_dim + j] * attention_weights[i];
            weighted_positive[j] = positive[i * embedding_dim + j] * attention_weights[i];
            weighted_negative[j] = negative[i * embedding_dim + j] * attention_weights[i];
        }

        // Calculate squared distances
        T distance_ap = 0.0f;
        T distance_an = 0.0f;

        for (int j = 0; j < embedding_dim; ++j) {
            distance_ap += (weighted_anchor[j] - weighted_positive[j]) * (weighted_anchor[j] - weighted_positive[j]);
            distance_an += (weighted_anchor[j] - weighted_negative[j]) * (weighted_anchor[j] - weighted_negative[j]);
        }

        // Apply margin ranking loss
        output[i] = (distance_an < distance_ap + 1.0f) ? 0.0f : distance_ap - distance_an + 1.0f;
    }
}

extern "C" {

void triplet_loss_with_attention(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* anchor = va_arg(args, const float*);
    int anchor_dim0 = va_arg(args, int);
    int anchor_dim1 = va_arg(args, int);

    const float* positive = va_arg(args, const float*);
    int positive_dim0 = va_arg(args, int);
    int positive_dim1 = va_arg(args, int);

    const float* negative = va_arg(args, const float*);
    int negative_dim0 = va_arg(args, int);
    int negative_dim1 = va_arg(args, int);

    const float* attention_weights = va_arg(args, const float*);
    int attention_weights_dim0 = va_arg(args, int);
    int attention_weights_dim1 = va_arg(args, int);

    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = anchor_dim0;
    int embedding_dim = anchor_dim1;

    // Allocate device memory
    float *d_anchor, *d_positive, *d_negative, *d_attention_weights, *d_output;
    cudaMalloc(&d_anchor, batch_size * embedding_dim * sizeof(float));
    cudaMalloc(&d_positive, batch_size * embedding_dim * sizeof(float));
    cudaMalloc(&d_negative, batch_size * embedding_dim * sizeof(float));
    cudaMalloc(&d_attention_weights, batch_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_anchor, anchor, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positive, positive, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative, negative, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attention_weights, attention_weights, batch_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    triplet_loss_with_attention_kernel<float><<<numBlocks, threadsPerBlock>>>(
        d_anchor, d_positive, d_negative, d_attention_weights, d_output, batch_size, embedding_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_anchor);
    cudaFree(d_positive);
    cudaFree(d_negative);
    cudaFree(d_attention_weights);
    cudaFree(d_output);
}

}  // extern "C"
