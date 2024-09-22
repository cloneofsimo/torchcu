
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end
#include <cutlass/cutlass.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for chain matrix multiplication and ReLU using bfloat16
__global__ void chain_matmul_relu_kernel_bf16(const float* input_tensor, const float* weight1, const float* weight2, float* output, 
                                        int m, int n, int k1, int k2) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k1; ++i) {
            for (int j = 0; j < k2; ++j) {
                __nv_bfloat16 a = float_to_bfloat16(input_tensor[row * k1 + i]);
                __nv_bfloat16 b = float_to_bfloat16(weight1[col * k1 + i]);
                __nv_bfloat16 c = float_to_bfloat16(weight2[j * k2 + i]); 
                sum += bfloat16_to_float(__hmul(__hmul(a, b), c)); 
            }
        }
        output[row * n + col] = fmaxf(sum, 0.0f);  // ReLU activation
    }
}

extern "C" {

void chain_matmul_bfloat16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    const float* weight1 = va_arg(args, const float*);
    int weight1_dim0 = va_arg(args, int);
    int weight1_dim1 = va_arg(args, int);

    const float* weight2 = va_arg(args, const float*);
    int weight2_dim0 = va_arg(args, int);
    int weight2_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int hidden_dim = weight1_dim0;
    int output_dim = weight2_dim0;

    // Allocate device memory
    float *d_input, *d_weight1, *d_weight2, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight1, hidden_dim * input_dim * sizeof(float));
    cudaMalloc(&d_weight2, output_dim * hidden_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight1, weight1, hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight2, weight2, output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    chain_matmul_relu_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight1, d_weight2, d_output, batch_size, output_dim, input_dim, hidden_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight1);
    cudaFree(d_weight2);
    cudaFree(d_output);
}

}  // extern "C"
