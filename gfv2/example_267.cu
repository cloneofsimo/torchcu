
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <stdarg.h>

#define WARP_SIZE 32

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
  return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
  return __half2float(h);
}

// Helper function to calculate dot product of two vectors with mixed precision (fp16 for accumulation)
__device__ __forceinline__ float dot_product_fp16(const half *a, const half *b, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += half_to_float(__hmul(a[i], b[i]));
    }
    return sum;
}

// CUDA kernel for cosine similarity with padding
__global__ void cosine_similarity_kernel(const int8_t *input_tensor, const int8_t *query_tensor, float *output,
                                        int input_tensor_dim0, int input_tensor_dim1,
                                        int query_tensor_dim0, int query_tensor_dim1,
                                        int padding_value) {
    // Calculate thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure thread indices are within bounds
    if (row < input_tensor_dim0 && col < input_tensor_dim1) {
        // Calculate indices for input and query tensors
        int input_index = row * input_tensor_dim1 + col;
        int query_index = row * query_tensor_dim1 + col;

        // Calculate dot product with mixed precision (fp16 for accumulation)
        float dot_product = dot_product_fp16(reinterpret_cast<const half*>(input_tensor + input_index),
                                               reinterpret_cast<const half*>(query_tensor + query_index),
                                               input_tensor_dim1);

        // Calculate magnitudes with mixed precision (fp16 for accumulation)
        float input_magnitude = sqrtf(dot_product_fp16(reinterpret_cast<const half*>(input_tensor + input_index),
                                                          reinterpret_cast<const half*>(input_tensor + input_index),
                                                          input_tensor_dim1));

        float query_magnitude = sqrtf(dot_product_fp16(reinterpret_cast<const half*>(query_tensor + query_index),
                                                          reinterpret_cast<const half*>(query_tensor + query_index),
                                                          query_tensor_dim1));

        // Calculate cosine similarity
        if (input_magnitude == 0.0f || query_magnitude == 0.0f) {
            output[col] = 0.0f;  // Handle cases where magnitude is zero to prevent division by zero
        } else {
            output[col] = dot_product / (input_magnitude * query_magnitude);
        }
    }
}

extern "C" {

void cosine_similarity_with_padding(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const int8_t *input_tensor = va_arg(args, const int8_t*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract query tensor
    const int8_t *query_tensor = va_arg(args, const int8_t*);
    int query_tensor_dim0 = va_arg(args, int);
    int query_tensor_dim1 = va_arg(args, int);

    // Extract padding value
    int padding_value = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float *output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    int8_t *d_input_tensor, *d_query_tensor;
    float *d_output;
    cudaMalloc(&d_input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(int8_t));
    cudaMalloc(&d_query_tensor, query_tensor_dim0 * query_tensor_dim1 * sizeof(int8_t));
    cudaMalloc(&d_output, input_tensor_dim1 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input_tensor, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_tensor, query_tensor, query_tensor_dim0 * query_tensor_dim1 * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(WARP_SIZE, 1);
    dim3 numBlocks((input_tensor_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);

    cosine_similarity_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input_tensor, d_query_tensor, d_output,
        input_tensor_dim0, input_tensor_dim1, query_tensor_dim0, query_tensor_dim1, padding_value
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input_tensor);
    cudaFree(d_query_tensor);
    cudaFree(d_output);
}
}
