
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

// CUDA kernel for variance calculation
__global__ void variance_kernel(const float* input, float* variance, int batch_size, int input_dim, int output_dim) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch < batch_size) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            float val = input[batch * input_dim + i];
            sum += val;
            sum_sq += val * val;
        }
        variance[batch] = (sum_sq - (sum * sum) / input_dim) / (input_dim - 1);
    }
}

// CUDA kernel for batched matrix multiplication and bias addition using bfloat16
__global__ void bmm_bias_kernel_bf16(const float* input, const float* weight, const float* bias, float* output,
                                    int batch_size, int input_dim, int output_dim, int weight_dim) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int output_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch < batch_size && output_idx < output_dim) {
        __nv_bfloat16 sum = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input[batch * input_dim + i]);
            __nv_bfloat16 b = float_to_bfloat16(weight[output_idx * weight_dim + i]);
            sum += __hmul(a, b);
        }
        output[batch * output_dim + output_idx] = bfloat16_to_float(sum) + bias[output_idx];
    }
}

extern "C" {

void expand_var_bmm_bf16_int8_function(int num_args, ...) {
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
    int weight_dim2 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);
    int bias_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    int8_t* output = va_arg(args, int8_t*);

    // Extract variance tensor (assuming it's preallocated)
    float* variance = va_arg(args, float*);

    va_end(args);

    int batch_size = weight_dim0;
    int input_dim = input_tensor_dim2;
    int output_dim = weight_dim2;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output, *d_variance;
    cudaMalloc(&d_input, batch_size * input_tensor_dim1 * input_tensor_dim2 * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * weight_dim2 * sizeof(float));
    cudaMalloc(&d_bias, bias_dim0 * bias_dim1 * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));
    cudaMalloc(&d_variance, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_tensor_dim1 * input_tensor_dim2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * weight_dim2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_dim0 * bias_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate variance
    dim3 threadsPerBlock_var(128);
    dim3 numBlocks_var((batch_size + threadsPerBlock_var.x - 1) / threadsPerBlock_var.x);
    variance_kernel<<<numBlocks_var, threadsPerBlock_var>>>(
        d_input, d_variance, batch_size, input_dim, input_tensor_dim1
    );

    // Launch kernel for batched matrix multiplication and bias addition
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, (output_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);
    bmm_bias_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output, batch_size, input_dim, output_dim, weight_dim1
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(variance, d_variance, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaFree(d_variance);
}

}  // extern "C"
