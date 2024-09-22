
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f); 
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for multi-margin loss with int8 precision
__global__ void multi_margin_loss_kernel_int8(const int8_t* input_tensor, const int8_t* target,
                                               const float* p, const float* margin, const char* reduction, 
                                               float* output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            if (i != target[row]) {
                float input_value = half_to_float(input_tensor[row * n + i]);
                float target_value = half_to_float(input_tensor[row * n + target[row]]);
                sum += fmaxf(0.0f, input_value - target_value + *margin);
            }
        }

        if (strcmp(reduction, "mean") == 0) {
            output[0] += sum / (float)m;
        } else if (strcmp(reduction, "sum") == 0) {
            output[0] += sum;
        } else if (strcmp(reduction, "none") == 0) {
            output[row] = sum;
        }
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const int8_t* input_tensor = va_arg(args, const int8_t*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract target tensor
    const int8_t* target = va_arg(args, const int8_t*);
    int target_dim0 = va_arg(args, int); 

    // Extract p value
    const float* p = va_arg(args, const float*);
    
    // Extract margin value
    const float* margin = va_arg(args, const float*);

    // Extract reduction string
    const char* reduction = va_arg(args, const char*);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    int8_t *d_input, *d_target;
    float *d_p, *d_margin, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(int8_t));
    cudaMalloc(&d_target, target_dim0 * sizeof(int8_t));
    cudaMalloc(&d_p, sizeof(float));
    cudaMalloc(&d_margin, sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, target_dim0 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_margin, margin, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_tensor_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_tensor_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    multi_margin_loss_kernel_int8<<<numBlocks, threadsPerBlock>>>(
        d_input, d_target, d_p, d_margin, reduction, d_output, input_tensor_dim0, input_tensor_dim1
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_p);
    cudaFree(d_margin);
    cudaFree(d_output);
}

}  // extern "C"
