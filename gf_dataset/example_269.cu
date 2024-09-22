
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

// CUDA kernel for rounding and max pooling
__global__ void round_maxpool_kernel(const float* input_tensor, half* rounded_tensor, half* pooled_tensor, 
                                     int batch_size, int channels, int D, int H, int W, int kernel_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int h = threadIdx.z;
    int w = threadIdx.y;

    if (b < batch_size && c < channels && d < D && h < H && w < W) {
        int input_idx = b * channels * D * H * W + c * D * H * W + d * H * W + h * W + w;
        rounded_tensor[input_idx] = float_to_half(roundf(input_tensor[input_idx]));

        // Max pooling (kernel_size is assumed to be odd)
        int kernel_half = kernel_size / 2;
        int max_val = -INFINITY;
        for (int kd = -kernel_half; kd <= kernel_half; kd++) {
            for (int kh = -kernel_half; kh <= kernel_half; kh++) {
                for (int kw = -kernel_half; kw <= kernel_half; kw++) {
                    int pool_d = d + kd;
                    int pool_h = h + kh;
                    int pool_w = w + kw;
                    if (pool_d >= 0 && pool_d < D && pool_h >= 0 && pool_h < H && pool_w >= 0 && pool_w < W) {
                        int pool_idx = b * channels * D * H * W + c * D * H * W + pool_d * H * W + pool_h * W + pool_w;
                        max_val = max(max_val, half_to_float(rounded_tensor[pool_idx]));
                    }
                }
            }
        }
        pooled_tensor[b * channels * D * H * W + c * D * H * W + d * H * W + h * W + w] = float_to_half(max_val);
    }
}

// CUDA kernel for top-k operation
__global__ void topk_kernel(const half* pooled_tensor, int* indices, float* values, int batch_size, int channels, int D, int H, int W, int k) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = threadIdx.y;

    if (b < batch_size && i < k) {
        // Flatten the pooled_tensor for each batch
        int flatten_idx = b * channels * D * H * W + i;
        int max_idx = 0;
        float max_val = -INFINITY;

        // Find the k-th largest value and its index
        for (int j = 0; j < channels * D * H * W; j++) {
            int idx = b * channels * D * H * W + j;
            float val = half_to_float(pooled_tensor[idx]);

            if (val > max_val) {
                max_val = val;
                max_idx = idx;
            }
        }

        // Store the index and value
        indices[flatten_idx] = max_idx;
        values[flatten_idx] = max_val;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);
    int input_tensor_dim4 = va_arg(args, int);

    // Extract kernel size
    const int kernel_size = va_arg(args, int);

    // Extract k
    const int k = va_arg(args, int);

    // Extract output tensors
    float* values = va_arg(args, float*);
    int* indices = va_arg(args, int*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int D = input_tensor_dim2;
    int H = input_tensor_dim3;
    int W = input_tensor_dim4;

    // Allocate device memory
    half *d_rounded_tensor, *d_pooled_tensor;
    cudaMalloc(&d_rounded_tensor, batch_size * channels * D * H * W * sizeof(half));
    cudaMalloc(&d_pooled_tensor, batch_size * channels * D * H * W * sizeof(half));

    // Allocate device memory for outputs
    float *d_values;
    int *d_indices;
    cudaMalloc(&d_values, batch_size * k * sizeof(float));
    cudaMalloc(&d_indices, batch_size * k * sizeof(int));

    // Launch kernels
    dim3 threadsPerBlock(32, 32, 4); // Adjust based on your hardware
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (channels + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (D + threadsPerBlock.z - 1) / threadsPerBlock.z);

    round_maxpool_kernel<<<numBlocks, threadsPerBlock>>>(
        input_tensor, d_rounded_tensor, d_pooled_tensor, batch_size, channels, D, H, W, kernel_size
    );

    topk_kernel<<<numBlocks, threadsPerBlock>>>(
        d_pooled_tensor, d_indices, d_values, batch_size, channels, D, H, W, k
    );

    // Copy results back to host
    cudaMemcpy(values, d_values, batch_size * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(indices, d_indices, batch_size * k * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_rounded_tensor);
    cudaFree(d_pooled_tensor);
    cudaFree(d_values);
    cudaFree(d_indices);
}

}  // extern "C"
