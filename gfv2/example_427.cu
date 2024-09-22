
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

// CUDA kernel for element-wise power, matrix multiplication, and bfloat16 conversion
__global__ void power_matmul_kernel_bf16(const float* input_tensor, const float* weight, 
                                        __nv_bfloat16* output, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            float input_val = input_tensor[row * k + i];
            sum += input_val * input_val * weight[col * k + i];  // Transposed access
        }
        output[row * n + col] = float_to_bfloat16(sum); 
    }
}

extern "C" {

void my_function(int num_args, ...) {
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

    // Extract output tensor (assuming it's preallocated)
    __nv_bfloat16* output = va_arg(args, __nv_bfloat16*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;

    // Allocate device memory
    float *d_input, *d_weight;
    __nv_bfloat16 *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(__nv_bfloat16));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    power_matmul_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, batch_size, output_dim, input_dim
    );

    // Copy result back to host (converting from bfloat16 to float)
    float* output_float = new float[batch_size * output_dim];
    cudaMemcpy(output_float, d_output, batch_size * output_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch_size * output_dim; ++i) {
        output[i] = float_to_bfloat16(output_float[i]); 
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    delete[] output_float;
}

}  // extern "C"
