
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>
#include <math.h>
#include <cutlass/cutlass.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for bucketing features
__global__ void bucketize_kernel(const float* input_tensor, int* bucket_indices, int num_features, int num_buckets) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_features) {
        float value = input_tensor[i];
        float bucket_size = 1.0f / num_buckets;
        bucket_indices[i] = (int)(value / bucket_size); 
    }
}

// CUDA kernel for calculating center loss per bucket
__global__ void center_loss_kernel(const float* features, const float* centers, float* bucket_losses,
                                   const int* bucket_indices, int num_features, int num_buckets) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_features) {
        int bucket = bucket_indices[i];
        if (bucket >= 0 && bucket < num_buckets) {
            float loss = 0.0f;
            for (int j = 0; j < 512; ++j) {
                float diff = features[i * 512 + j] - centers[bucket * 512 + j];
                loss += diff * diff;
            }
            atomicAdd(&bucket_losses[bucket], loss);
        }
    }
}

// CUDA kernel for weighted sum of bucket losses
__global__ void weighted_sum_kernel(const float* bucket_losses, const float* weights, float* total_loss, int num_buckets) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_buckets) {
        atomicAdd(total_loss, bucket_losses[i] * weights[i]);
    }
}

extern "C" {

void torch_center_loss_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* features = va_arg(args, const float*);
    int num_features = va_arg(args, int);
    int feature_dim = va_arg(args, int);

    const float* centers = va_arg(args, const float*);
    int num_centers = va_arg(args, int);
    int center_dim = va_arg(args, int);

    const float* weights = va_arg(args, const float*);
    int num_weights = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* total_loss = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    int* d_bucket_indices;
    float* d_features, *d_centers, *d_weights, *d_bucket_losses;
    cudaMalloc(&d_bucket_indices, num_features * sizeof(int));
    cudaMalloc(&d_features, num_features * feature_dim * sizeof(float));
    cudaMalloc(&d_centers, num_centers * center_dim * sizeof(float));
    cudaMalloc(&d_weights, num_weights * sizeof(float));
    cudaMalloc(&d_bucket_losses, num_centers * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_features, features, num_features * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centers, centers, num_centers * center_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, num_weights * sizeof(float), cudaMemcpyHostToDevice);

    // Bucketize features
    int num_buckets = 10;
    bucketize_kernel<<<(num_features + 255) / 256, 256>>>(d_features, d_bucket_indices, num_features, num_buckets);

    // Calculate center loss per bucket
    center_loss_kernel<<<(num_features + 255) / 256, 256>>>(d_features, d_centers, d_bucket_losses,
                                                          d_bucket_indices, num_features, num_buckets);

    // Weighted sum of bucket losses
    weighted_sum_kernel<<<(num_buckets + 255) / 256, 256>>>(d_bucket_losses, d_weights, total_loss, num_buckets);

    // Copy result back to host
    cudaMemcpy(total_loss, total_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_bucket_indices);
    cudaFree(d_features);
    cudaFree(d_centers);
    cudaFree(d_weights);
    cudaFree(d_bucket_losses);
}

}  // extern "C"
