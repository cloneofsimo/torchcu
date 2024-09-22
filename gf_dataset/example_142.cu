
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for batch matrix multiplication and ReLU using bfloat16
__global__ void bmm_relu_kernel_bf16(const float* input_tensor, const float* weight, float* output, 
                                        int batch_size, int m, int k, int n) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input_tensor[(batch_idx * m * k) + (row * k) + i]);
            __nv_bfloat16 b = float_to_bfloat16(weight[(batch_idx * k * n) + (i * n) + col]); 
            sum += bfloat16_to_float(__hmul(a, b));
        }
        output[(batch_idx * m * n) + (row * n) + col] = fmaxf(sum, 0.0f);
    }
}

extern "C" {

void torch_function(int num_args, ...) {
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

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int m = input_tensor_dim1;
    int k = input_tensor_dim2;
    int n = weight_dim2;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, batch_size * m * k * sizeof(float));
    cudaMalloc(&d_weight, batch_size * k * n * sizeof(float));
    cudaMalloc(&d_output, batch_size * m * n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, batch_size * k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    bmm_relu_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, batch_size, m, k, n
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
