
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

__global__ void round_local_attention_mean_int8_kernel(const int8_t* input, const int8_t* query, const int8_t* key, const int8_t* value, 
                                        int batch_size, int seq_len, int head_size, float* output) {

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && h < head_size) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            int input_idx = b * seq_len * head_size + i * head_size + h;
            int query_idx = b * seq_len * head_size + i * head_size + h;
            int key_idx = b * seq_len * head_size + i * head_size + h;
            int value_idx = b * seq_len * head_size + i * head_size + h;

            float attention_score = (float)input[input_idx] * (float)query[query_idx] * (float)key[key_idx] * (float)value[value_idx];
            sum += attention_score;
        }

        // Round and mean pooling
        sum = roundf(sum / seq_len);
        output[b * head_size + h] = sum;
    }
}

extern "C" {

void round_local_attention_mean_int8(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    const float* query_tensor = va_arg(args, const float*);
    int query_tensor_dim0 = va_arg(args, int);
    int query_tensor_dim1 = va_arg(args, int);

    const float* key_tensor = va_arg(args, const float*);
    int key_tensor_dim0 = va_arg(args, int);
    int key_tensor_dim1 = va_arg(args, int);

    const float* value_tensor = va_arg(args, const float*);
    int value_tensor_dim0 = va_arg(args, int);
    int value_tensor_dim1 = va_arg(args, int);

    // Extract window_size
    int window_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Dimensions
    int batch_size = input_tensor_dim0;
    int seq_len = input_tensor_dim1;
    int head_size = input_tensor_dim1 / window_size;

    // Allocate device memory
    int8_t *d_input, *d_query, *d_key, *d_value;
    float *d_output;
    cudaMalloc(&d_input, batch_size * seq_len * head_size * sizeof(int8_t));
    cudaMalloc(&d_query, batch_size * seq_len * head_size * sizeof(int8_t));
    cudaMalloc(&d_key, batch_size * seq_len * head_size * sizeof(int8_t));
    cudaMalloc(&d_value, batch_size * seq_len * head_size * sizeof(int8_t));
    cudaMalloc(&d_output, batch_size * head_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * seq_len * head_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query_tensor, batch_size * seq_len * head_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key_tensor, batch_size * seq_len * head_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value_tensor, batch_size * seq_len * head_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 8);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, (head_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    round_local_attention_mean_int8_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_query, d_key, d_value, batch_size, seq_len, head_size, d_output
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * head_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
}

}  // extern "C"
