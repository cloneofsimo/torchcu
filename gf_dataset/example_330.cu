
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

#include <cutlass/cutlass.h>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for matrix multiplication and ReLU using bfloat16
__global__ void matmul_bce_with_logits_kernel_fp16(const float* input_tensor, const float* target, const float* weight, float* output, 
                                        int batch_size, int input_dim, int output_dim, int target_dim) {
    int batch_idx = blockIdx.x;
    int row = threadIdx.x;
    int col = threadIdx.y;
    
    if (batch_idx < batch_size && row < input_dim && col < output_dim) {
        float sum = 0.0f;
        for (int i = 0; i < target_dim; ++i) {
            half a = float_to_half(input_tensor[batch_idx * input_dim * target_dim + row * target_dim + i]);
            half b = float_to_half(weight[col * target_dim + i]);  // Transposed access
            sum += half_to_float(__hmul(a, b));
        }
        output[batch_idx * input_dim * output_dim + row * output_dim + col] = sum;
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

    // Extract target tensor
    const float* target = va_arg(args, const float*);
    int target_dim0 = va_arg(args, int);
    int target_dim1 = va_arg(args, int);
    int target_dim2 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;
    int target_dim = input_tensor_dim2;

    // Allocate device memory
    float *d_input, *d_target, *d_weight, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * target_dim * sizeof(float));
    cudaMalloc(&d_target, batch_size * input_dim * target_dim * sizeof(float));
    cudaMalloc(&d_weight, output_dim * target_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * target_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, batch_size * input_dim * target_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * target_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(batch_size, 1);
    matmul_bce_with_logits_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_target, d_weight, d_output, batch_size, input_dim, output_dim, target_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_dim * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
