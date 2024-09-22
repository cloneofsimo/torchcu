
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

// CUDA kernel for transformer encoder with int8 and bfloat16
__global__ void transformer_encoder_kernel_int8_bf16(const float* input_tensor, const float* weight, const float* bias, 
                                                 float* output, int batch_size, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * seq_len * dim) {
        int b = idx / (seq_len * dim);
        int s = (idx % (seq_len * dim)) / dim;
        int d = idx % dim;

        __nv_bfloat16 sum = 0.0f;
        for (int i = 0; i < dim; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input_tensor[b * seq_len * dim + s * dim + i]);
            __nv_bfloat16 w = float_to_bfloat16(weight[d * dim + i]);
            sum += __hmul(a, w);
        }
        sum = __hmul(sum, float_to_bfloat16(1.0f / (float)dim));
        sum += float_to_bfloat16(bias[d]);
        output[b * seq_len * dim + s * dim + d] = bfloat16_to_float(__hmul(sum, sum)); // Relu with int8
        output[b * seq_len * dim + s * dim + d] = bfloat16_to_float(__hmul(sum, sum)); // Relu with int8
        output[b * seq_len * dim + s * dim + d] = bfloat16_to_float(__hmul(sum, sum)); // Relu with int8
    }
}

extern "C" {

void transformer_encoder_int8_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int seq_len = input_tensor_dim1;
    int dim = input_tensor_dim2;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * seq_len * dim * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * sizeof(float));
    cudaMalloc(&d_bias, bias_dim0 * sizeof(float));
    cudaMalloc(&d_output, batch_size * seq_len * dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * seq_len * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size * seq_len * dim + threadsPerBlock.x - 1) / threadsPerBlock.x);

    transformer_encoder_kernel_int8_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output, batch_size, seq_len, dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * seq_len * dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
