
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end
#include <math.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for pairwise Euclidean distance
__global__ void pairwise_distance_kernel_bf16(const float* input, const float* other, float* distances, int batch_size, int embedding_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < batch_size && j < batch_size) {
        float sum = 0.0f;
        for (int k = 0; k < embedding_dim; ++k) {
            __nv_bfloat16 a = float_to_bfloat16(input[i * embedding_dim + k]);
            __nv_bfloat16 b = float_to_bfloat16(other[j * embedding_dim + k]);
            sum += bfloat16_to_float(__hmul(a - b, a - b));
        }
        distances[i * batch_size + j] = sqrtf(sum);
    }
}

// CUDA kernel for hinge embedding loss calculation
__global__ void hinge_loss_kernel_bf16(const float* distances, const int* targets, float* loss, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        float positive_loss = 0.0f;
        int positive_count = 0;
        float negative_loss = 0.0f;
        int negative_count = 0;
        for (int j = 0; j < batch_size; ++j) {
            if (targets[i] == targets[j]) {
                positive_loss += distances[i * batch_size + j];
                positive_count++;
            } else {
                negative_loss += distances[i * batch_size + j];
                negative_count++;
            }
        }

        if (positive_count > 0) {
            positive_loss /= (float)positive_count;
        }
        if (negative_count > 0) {
            negative_loss /= (float)negative_count;
        }

        loss[i] = fmaxf(1.0f - negative_loss + positive_loss, 0.0f);
    }
}

extern "C" {

void embedding_loss_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract target tensor
    const int* target_tensor = va_arg(args, const int*);
    int target_tensor_dim0 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int embedding_dim = input_tensor_dim1;

    // Allocate device memory
    float* d_input, *d_distances, *d_loss;
    int* d_targets;
    cudaMalloc(&d_input, batch_size * embedding_dim * sizeof(float));
    cudaMalloc(&d_distances, batch_size * batch_size * sizeof(float));
    cudaMalloc(&d_loss, batch_size * sizeof(float));
    cudaMalloc(&d_targets, batch_size * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, target_tensor, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate pairwise distances
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    pairwise_distance_kernel_bf16<<<numBlocks, threadsPerBlock>>>(d_input, d_input, d_distances, batch_size, embedding_dim);

    // Calculate hinge loss
    numBlocks = (batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    hinge_loss_kernel_bf16<<<numBlocks, threadsPerBlock>>>(d_distances, d_targets, d_loss, batch_size);

    // Reduce loss to a single value (sum)
    float* d_loss_sum;
    cudaMalloc(&d_loss_sum, sizeof(float));
    cudaMemset(d_loss_sum, 0, sizeof(float));
    cudaMemcpyAsync(d_loss_sum, d_loss, sizeof(float), cudaMemcpyDeviceToDevice);

    // Calculate sum on device
    float* d_loss_sum_temp = d_loss_sum;
    for (int i = 1; i < batch_size; i++) {
        cudaMemcpyAsync(d_loss_sum_temp + 1, d_loss + i, sizeof(float), cudaMemcpyDeviceToDevice);
        d_loss_sum_temp += 1;
        cudaMemcpyAsync(d_loss_sum, d_loss_sum + 1, sizeof(float), cudaMemcpyDeviceToDevice);
        d_loss_sum += 1;
    }

    // Copy sum back to host
    cudaMemcpy(output, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_distances);
    cudaFree(d_loss);
    cudaFree(d_targets);
    cudaFree(d_loss_sum);
}
}  // extern "C"
