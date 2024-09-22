
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for contrastive loss calculation
__global__ void contrastive_loss_kernel(const float* anchor_features, const float* positive_features, 
                                     const float* negative_features, const float* weight, 
                                     float* loss_contrastive, float* sparsity_loss,
                                     int batch_size, int feature_dim, int num_negatives, 
                                     float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        // Calculate similarity scores
        float dot_positive = 0.0f;
        for (int j = 0; j < feature_dim; ++j) {
            dot_positive += anchor_features[i * feature_dim + j] * positive_features[i * feature_dim + j];
        }

        float dot_negatives[10]; // Assuming max 10 negatives
        for (int k = 0; k < num_negatives; ++k) {
            dot_negatives[k] = 0.0f;
            for (int j = 0; j < feature_dim; ++j) {
                dot_negatives[k] += anchor_features[i * feature_dim + j] * negative_features[(i * num_negatives + k) * feature_dim + j];
            }
        }

        // Apply temperature scaling
        float similarity_scores[11];
        similarity_scores[0] = dot_positive / temperature;
        for (int k = 0; k < num_negatives; ++k) {
            similarity_scores[k + 1] = dot_negatives[k] / temperature;
        }

        // Calculate contrastive loss
        float loss_local = 0.0f;
        for (int k = 1; k < 11; ++k) {
            loss_local += expf(similarity_scores[k]); 
        }
        loss_local = logf(1.0f + loss_local / expf(similarity_scores[0]));
        loss_contrastive[i] = loss_local;

        // Calculate sparsity loss
        float sparsity_local = 0.0f;
        for (int j = 0; j < feature_dim; ++j) {
            sparsity_local += fabsf(weight[j]);
        }
        sparsity_loss[i] = sparsity_local / feature_dim;
    }
}

extern "C" {

void contrastive_loss_with_sparsity(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* anchor_features = va_arg(args, const float*);
    int anchor_features_dim0 = va_arg(args, int);
    int anchor_features_dim1 = va_arg(args, int);

    const float* positive_features = va_arg(args, const float*);
    int positive_features_dim0 = va_arg(args, int);
    int positive_features_dim1 = va_arg(args, int);

    const float* negative_features = va_arg(args, const float*);
    int negative_features_dim0 = va_arg(args, int);
    int negative_features_dim1 = va_arg(args, int);
    int negative_features_dim2 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);

    // Extract output tensors (assuming they're preallocated)
    float* loss_contrastive = va_arg(args, float*);
    float* sparsity_loss = va_arg(args, float*);

    va_end(args);

    int batch_size = anchor_features_dim0;
    int feature_dim = anchor_features_dim1;
    int num_negatives = negative_features_dim1;
    float temperature = 0.1f;

    // Allocate device memory
    float* d_anchor_features, *d_positive_features, *d_negative_features, *d_weight;
    float* d_loss_contrastive, *d_sparsity_loss;
    cudaMalloc(&d_anchor_features, batch_size * feature_dim * sizeof(float));
    cudaMalloc(&d_positive_features, batch_size * feature_dim * sizeof(float));
    cudaMalloc(&d_negative_features, batch_size * num_negatives * feature_dim * sizeof(float));
    cudaMalloc(&d_weight, feature_dim * sizeof(float));
    cudaMalloc(&d_loss_contrastive, batch_size * sizeof(float));
    cudaMalloc(&d_sparsity_loss, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_anchor_features, anchor_features, batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positive_features, positive_features, batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative_features, negative_features, batch_size * num_negatives * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, feature_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    contrastive_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_anchor_features, d_positive_features, d_negative_features, d_weight,
        d_loss_contrastive, d_sparsity_loss, batch_size, feature_dim, num_negatives, temperature
    );

    // Copy result back to host
    cudaMemcpy(loss_contrastive, d_loss_contrastive, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sparsity_loss, d_sparsity_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_anchor_features);
    cudaFree(d_positive_features);
    cudaFree(d_negative_features);
    cudaFree(d_weight);
    cudaFree(d_loss_contrastive);
    cudaFree(d_sparsity_loss);
}

}  // extern "C"
