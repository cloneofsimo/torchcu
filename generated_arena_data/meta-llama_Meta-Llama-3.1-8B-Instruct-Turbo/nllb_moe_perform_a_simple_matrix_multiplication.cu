
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

// CUDA kernel for matrix multiplication with ReLU activation and bias addition using bfloat16
__global__ void matmul_relu_kernel_bf16(const float* input_tensor, const float* weight, const float* bias, float* output, 
                                            int m, int n, int k, int bias_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input_tensor[row * k + i]);
            __nv_bfloat16 b = float_to_bfloat16(weight[col * k + i]);  // Transposed access
            sum += bfloat16_to_float(__hmul(a, b));
        }
        sum += bias[col];
        output[row * n + col] = fmaxf(sum, 0.0f);  // ReLU activation
    }
}

extern "C" {

void matmul_with_relu_function(int num_args, ...) {
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

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_bias, output_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_relu_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output, batch_size, output_dim, input_dim, bias_dim0
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
