
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

// CUDA kernel for fused dropout, dot product, and softshrink
__global__ void fused_kernel(const float* input_tensor, const float* weight, const float* bias, 
                             float* output, int batch_size, int input_size, int output_size,
                             float dropout_p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input_tensor[row * input_size + i] * weight[col * input_size + i];
        }
        sum += bias[col];

        // Apply dropout
        if (rand() / RAND_MAX > dropout_p) {
            sum = 0.0f;  // Set to 0 if dropout occurs
        }

        // Apply softshrink
        if (sum < -1.0f) {
            sum = sum + 1.0f;
        } else if (sum > 1.0f) {
            sum = sum - 1.0f;
        } else {
            sum = 0.0f;
        }

        output[row * output_size + col] = sum;
    }
}

extern "C" {

void fused_dropout_dot_softshrink(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    // Extract dropout probability
    float dropout_p = va_arg(args, float);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_size = input_tensor_dim1;
    int output_size = weight_dim0;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_weight, output_size * input_size * sizeof(float));
    cudaMalloc(&d_bias, output_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    fused_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output, batch_size, input_size, output_size, dropout_p
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
