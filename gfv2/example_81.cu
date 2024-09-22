
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

__global__ void cosine_embedding_loss_kernel(const float* input1, const float* input2, const int* target, 
                                             float margin, float* output, int batch_size, int feature_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float dot_product = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        for (int i = 0; i < feature_dim; ++i) {
            dot_product += input1[idx * feature_dim + i] * input2[idx * feature_dim + i];
            norm1 += input1[idx * feature_dim + i] * input1[idx * feature_dim + i];
            norm2 += input2[idx * feature_dim + i] * input2[idx * feature_dim + i];
        }
        float cosine_similarity = dot_product / (sqrtf(norm1) * sqrtf(norm2));
        float loss = fmaxf(margin - target[idx] * cosine_similarity, 0.0f);
        output[idx] = loss;
    }
}

__global__ void reduction_kernel(const float* loss, float* output, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        output[0] += loss[idx];
    }
}

extern "C" {
void cosine_embedding_loss_example(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input1 = va_arg(args, const float*);
    int input1_dim0 = va_arg(args, int);
    int input1_dim1 = va_arg(args, int);

    const float* input2 = va_arg(args, const float*);
    int input2_dim0 = va_arg(args, int);
    int input2_dim1 = va_arg(args, int);

    // Extract target tensor
    const int* target = va_arg(args, const int*);
    int target_dim0 = va_arg(args, int);

    // Extract margin
    float margin = va_arg(args, float);

    // Extract reduction (not used in this implementation)
    const char* reduction = va_arg(args, const char*);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input1_dim0;
    int feature_dim = input1_dim1;

    // Allocate device memory
    float *d_input1, *d_input2, *d_loss, *d_output;
    cudaMalloc(&d_input1, batch_size * feature_dim * sizeof(float));
    cudaMalloc(&d_input2, batch_size * feature_dim * sizeof(float));
    cudaMalloc(&d_loss, batch_size * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input1, input1, batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2, batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_loss, target, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel for cosine embedding loss calculation
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    cosine_embedding_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input1, d_input2, d_loss, margin, d_loss, batch_size, feature_dim
    );

    // Launch kernel for reduction (mean)
    reduction_kernel<<<1, 256>>>(d_loss, d_output, batch_size);

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_loss);
    cudaFree(d_output);
}
}
