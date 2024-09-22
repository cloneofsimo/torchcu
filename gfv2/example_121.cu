
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to __half
__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert __half to float
__device__ __forceinline__ float half_to_float(__half h) {
    return __half2float(h);
}

// CUDA kernel for multi-scale attention with fp16
__global__ void multi_scale_attention_kernel_fp16(const float* input_tensor, const float* query_tensor,
                                               const float* key_tensor, const float* value_tensor, 
                                               const int* scales, float* output, int batch_size, 
                                               int seq_len, int dim, int num_scales) {
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && t < seq_len) {
        __half sum = 0.0h;

        for (int s = 0; s < num_scales; ++s) {
            // Calculate attention scores
            __half score = 0.0h;
            for (int d = 0; d < dim; ++d) {
                __half q = float_to_half(query_tensor[b * seq_len * dim + t * dim + d]);
                __half k = float_to_half(key_tensor[b * seq_len * dim + t * dim + d]);
                score += q * k;
            }

            // Apply scaling to attention scores
            score = score / (float_to_half(dim) * float_to_half(scales[s]));

            // Apply softmax (unnormalized)
            __half exp_score = exp(score);

            // Calculate context vector (unnormalized)
            for (int d = 0; d < dim; ++d) {
                __half v = float_to_half(value_tensor[b * seq_len * dim + t * dim + d]);
                sum += exp_score * v;
            }
        }

        // Store the final output
        output[b * seq_len * dim + t * dim] = half_to_float(sum);
    }
}

extern "C" {

void multi_scale_attention_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    const float* query_tensor = va_arg(args, const float*);
    int query_tensor_dim0 = va_arg(args, int);
    int query_tensor_dim1 = va_arg(args, int);
    int query_tensor_dim2 = va_arg(args, int);

    const float* key_tensor = va_arg(args, const float*);
    int key_tensor_dim0 = va_arg(args, int);
    int key_tensor_dim1 = va_arg(args, int);
    int key_tensor_dim2 = va_arg(args, int);

    const float* value_tensor = va_arg(args, const float*);
    int value_tensor_dim0 = va_arg(args, int);
    int value_tensor_dim1 = va_arg(args, int);
    int value_tensor_dim2 = va_arg(args, int);

    // Extract scales
    const int* scales = va_arg(args, const int*);
    int num_scales = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int seq_len = input_tensor_dim1;
    int dim = input_tensor_dim2;

    // Allocate device memory
    float *d_input, *d_query, *d_key, *d_value, *d_output;
    cudaMalloc(&d_input, batch_size * seq_len * dim * sizeof(float));
    cudaMalloc(&d_query, batch_size * seq_len * dim * sizeof(float));
    cudaMalloc(&d_key, batch_size * seq_len * dim * sizeof(float));
    cudaMalloc(&d_value, batch_size * seq_len * dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * seq_len * dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * seq_len * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query_tensor, batch_size * seq_len * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key_tensor, batch_size * seq_len * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value_tensor, batch_size * seq_len * dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    multi_scale_attention_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_query, d_key, d_value, scales, d_output, batch_size, seq_len, dim, num_scales
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * seq_len * dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
}

}  // extern "C"
