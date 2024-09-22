
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for self-attention with int8 quantization
__global__ void self_attention_int8_kernel(const int* input_tensor, const int* query_weight, const int* key_weight, 
                                          const int* value_weight, float* output, int batch_size, int seq_len, int hidden_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < seq_len) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            half q = float_to_half((float)input_tensor[row * seq_len * hidden_size + col * hidden_size + i]);
            half k = float_to_half((float)query_weight[i * hidden_size + i]);
            half v = float_to_half((float)key_weight[i * hidden_size + i]);

            // Calculate attention scores
            half attention_score = __expf(__fmaf(q, k, -0.5f * logf((float)hidden_size))); // exp(q*k - 0.5 * log(hidden_size))

            // Apply value weights
            v = float_to_half((float)value_weight[i * hidden_size + i]);
            sum += half_to_float(attention_score * v);
        }
        output[row * seq_len * hidden_size + col * hidden_size] = sum;
    }
}

extern "C" {

void self_attention_int8(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const int* input_tensor = va_arg(args, const int*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract query weight tensor
    const int* query_weight = va_arg(args, const int*);
    int query_weight_dim0 = va_arg(args, int);
    int query_weight_dim1 = va_arg(args, int);

    // Extract key weight tensor
    const int* key_weight = va_arg(args, const int*);
    int key_weight_dim0 = va_arg(args, int);
    int key_weight_dim1 = va_arg(args, int);

    // Extract value weight tensor
    const int* value_weight = va_arg(args, const int*);
    int value_weight_dim0 = va_arg(args, int);
    int value_weight_dim1 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int seq_len = input_tensor_dim1;
    int hidden_size = input_tensor_dim2;

    // Allocate device memory
    int *d_input, *d_query_weight, *d_key_weight, *d_value_weight;
    float *d_output;
    cudaMalloc(&d_input, batch_size * seq_len * hidden_size * sizeof(int));
    cudaMalloc(&d_query_weight, query_weight_dim0 * query_weight_dim1 * sizeof(int));
    cudaMalloc(&d_key_weight, key_weight_dim0 * key_weight_dim1 * sizeof(int));
    cudaMalloc(&d_value_weight, value_weight_dim0 * value_weight_dim1 * sizeof(int));
    cudaMalloc(&d_output, batch_size * seq_len * hidden_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * seq_len * hidden_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_weight, query_weight, query_weight_dim0 * query_weight_dim1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key_weight, key_weight, key_weight_dim0 * key_weight_dim1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_weight, value_weight, value_weight_dim0 * value_weight_dim1 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    self_attention_int8_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_query_weight, d_key_weight, d_value_weight, d_output, batch_size, seq_len, hidden_size
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_query_weight);
    cudaFree(d_key_weight);
    cudaFree(d_value_weight);
    cudaFree(d_output);
}

}  // extern "C"
