
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for pairwise Manhattan distance calculation (bfloat16)
__global__ void manhattan_distance_kernel_bf16(const float* input_tensor, const float* other_tensor,
                                                float* distances, int batch_size, int num_features, int seq_len) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int seq_idx = threadIdx.z;

    if (batch_idx < batch_size && feature_idx < num_features && seq_idx < seq_len) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input_tensor[batch_idx * num_features * seq_len + feature_idx * seq_len + i]);
            __nv_bfloat16 b = float_to_bfloat16(other_tensor[batch_idx * num_features * seq_len + feature_idx * seq_len + i]);
            sum += bfloat16_to_float(abs(__hsub(a, b)));
        }
        distances[batch_idx * num_features * seq_len + feature_idx * seq_len + seq_idx] = sum;
    }
}

// CUDA kernel for adaptive max pooling along the sequence dimension (bfloat16)
__global__ void adaptive_max_pool1d_kernel_bf16(const float* distances, float* output,
                                                 int batch_size, int num_features, int seq_len) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch_size && feature_idx < num_features) {
        float max_val = -FLT_MAX;  // Initialize with negative infinity
        for (int i = 0; i < seq_len; ++i) {
            __nv_bfloat16 val = float_to_bfloat16(distances[batch_idx * num_features * seq_len + feature_idx * seq_len + i]);
            if (bfloat16_to_float(val) > max_val) {
                max_val = bfloat16_to_float(val);
            }
        }
        output[batch_idx * num_features + feature_idx] = max_val;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    const float* other_tensor = va_arg(args, const float*);
    int other_tensor_dim0 = va_arg(args, int);
    int other_tensor_dim1 = va_arg(args, int);
    int other_tensor_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int num_features = input_tensor_dim1;
    int seq_len = input_tensor_dim2;

    // Allocate device memory
    float *d_input, *d_other, *d_distances, *d_output;
    cudaMalloc(&d_input, batch_size * num_features * seq_len * sizeof(float));
    cudaMalloc(&d_other, batch_size * num_features * seq_len * sizeof(float));
    cudaMalloc(&d_distances, batch_size * num_features * seq_len * sizeof(float));
    cudaMalloc(&d_output, batch_size * num_features * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * num_features * seq_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_other, other_tensor, batch_size * num_features * seq_len * sizeof(float), cudaMemcpyHostToDevice);

    // Launch Manhattan distance kernel
    dim3 threadsPerBlock(16, 16, 4);  // 16x16 threads per block, 4 threads per block for sequence dimension
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_features + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

    manhattan_distance_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_other, d_distances, batch_size, num_features, seq_len
    );

    // Launch adaptive max pooling kernel
    dim3 pool_threadsPerBlock(16, 16);
    dim3 pool_numBlocks((batch_size + pool_threadsPerBlock.x - 1) / pool_threadsPerBlock.x,
                        (num_features + pool_threadsPerBlock.y - 1) / pool_threadsPerBlock.y);

    adaptive_max_pool1d_kernel_bf16<<<pool_numBlocks, pool_threadsPerBlock>>>(
        d_distances, d_output, batch_size, num_features, seq_len
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * num_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_other);
    cudaFree(d_distances);
    cudaFree(d_output);
}

}  // extern "C"
