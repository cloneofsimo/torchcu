
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

#include <cutlass/cutlass.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for matrix multiplication and ReLU using bfloat16
__global__ void matmul_bf16_kernel(const float* input_tensor, const float* weight, float* output,
                                        int batch_size, int input_dim, int output_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < output_dim) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input_tensor[row * input_dim + i]);
            __nv_bfloat16 b = float_to_bfloat16(weight[col * input_dim + i]);  // Transposed access
            sum += bfloat16_to_float(__hmul(a, b));
        }
        output[row * output_dim + col] = sum;
    }
}

extern "C" {

void permute_round_grad_accumulate_bf16(int num_args, ...) {
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

    // Extract scale
    float scale = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * input_tensor_dim2 * sizeof(float));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * input_tensor_dim2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_bf16_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, batch_size, input_dim, output_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
