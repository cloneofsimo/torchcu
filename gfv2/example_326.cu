
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// Helper functions for FP16 conversions
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f); 
}
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

__global__ void attention_pow_fp16_kernel(const float* query, const float* key, const float* value, float power, float* output,
                                        int batch_size, int seq_len, int hidden_dim) {
    int b = blockIdx.z * blockDim.z + threadIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && i < seq_len && j < seq_len) {
        float sum = 0.0f;
        for (int k = 0; k < hidden_dim; ++k) {
            half q = float_to_half(query[b * seq_len * hidden_dim + i * hidden_dim + k]);
            half k_t = float_to_half(key[b * seq_len * hidden_dim + j * hidden_dim + k]);
            sum += half_to_float(__hmul(q, k_t)); 
        }
        float score = powf(sum, power);

        for (int k = 0; k < hidden_dim; ++k) {
            half v = float_to_half(value[b * seq_len * hidden_dim + j * hidden_dim + k]);
            sum += half_to_float(__hmul(float_to_half(score), v));
        }

        output[b * seq_len * hidden_dim + i * hidden_dim + j] = sum; 
    }
}


extern "C" {
void attention_pow_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* query = va_arg(args, const float*);
    int query_dim0 = va_arg(args, int);
    int query_dim1 = va_arg(args, int);
    int query_dim2 = va_arg(args, int);

    const float* key = va_arg(args, const float*);
    int key_dim0 = va_arg(args, int);
    int key_dim1 = va_arg(args, int);
    int key_dim2 = va_arg(args, int);

    const float* value = va_arg(args, const float*);
    int value_dim0 = va_arg(args, int);
    int value_dim1 = va_arg(args, int);
    int value_dim2 = va_arg(args, int);

    float power = va_arg(args, float);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    // Assert dimensions match
    assert(query_dim0 == key_dim0 && query_dim0 == value_dim0);
    assert(query_dim1 == key_dim1 && query_dim1 == value_dim1);
    assert(query_dim2 == key_dim2 && query_dim2 == value_dim2);

    int batch_size = query_dim0;
    int seq_len = query_dim1;
    int hidden_dim = query_dim2;

    // Allocate device memory
    float *d_query, *d_key, *d_value, *d_output;
    cudaMalloc(&d_query, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_key, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_value, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * seq_len * seq_len * hidden_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_query, query, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);
    
    attention_pow_fp16_kernel<<<numBlocks, threadsPerBlock>>>(
        d_query, d_key, d_value, power, d_output, batch_size, seq_len, hidden_dim
    );
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * seq_len * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
}

}  // extern "C"
