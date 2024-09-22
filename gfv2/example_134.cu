
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// Swiglu activation function
__device__ __forceinline__ float swiglu(float x) {
    return x * 1.0f / (1.0f + expf(-x));
}

// Kernel for matrix multiplication and Swiglu activation
__global__ void matmul_swiglu_kernel_bf16(const float* input_tensor, const float* weight, float* output, 
                                            int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input_tensor[row * k + i]);
            __nv_bfloat16 b = float_to_bfloat16(weight[col * k + i]);
            sum += bfloat16_to_float(__hmul(a, b));
        }
        output[row * n + col] = swiglu(sum);
    }
}

// Kernel for channel attention
__global__ void channel_attention_kernel(const float* output, const float* channel_attention_weight, float* result,
                                        int batch_size, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * output_dim) {
        int batch = idx / output_dim;
        int feature = idx % output_dim;
        result[idx] = output[idx] * channel_attention_weight[batch];
    }
}

// Kernel for calculating the ArcFace loss
__global__ void arcface_loss_kernel(const float* output, const int* target, float* loss, int batch_size, 
                                      int output_dim, float s, float m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        int label = target[idx];
        float max_score = -FLT_MAX;
        for (int j = 0; j < output_dim; ++j) {
            if (j == label) {
                max_score = output[idx * output_dim + j];
            }
        }
        
        float phi = max_score * s;
        phi = phi + m;
        
        loss[idx] = -logf(expf(phi) / (expf(phi) + expf(output[idx * output_dim + label])));
    }
}

extern "C" {

void forward_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weight tensor
    const float* weight_tensor = va_arg(args, const float*);
    int weight_tensor_dim0 = va_arg(args, int);
    int weight_tensor_dim1 = va_arg(args, int);

    // Extract target tensor
    const int* target = va_arg(args, const int*);
    int target_dim = va_arg(args, int);

    // Extract channel attention weight tensor
    const float* channel_attention_weight = va_arg(args, const float*);
    int channel_attention_weight_dim0 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);
    
    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_tensor_dim0;

    // Allocate device memory
    float *d_input, *d_weight, *d_output, *d_channel_attention_weight;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));
    cudaMalloc(&d_channel_attention_weight, channel_attention_weight_dim0 * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight_tensor, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_channel_attention_weight, channel_attention_weight, channel_attention_weight_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch matrix multiplication and Swiglu kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matmul_swiglu_kernel_bf16<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_output, batch_size, output_dim, input_dim);

    // Launch channel attention kernel
    numBlocks = (batch_size * output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x;
    channel_attention_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_channel_attention_weight, d_output, batch_size, output_dim);

    // Calculate ArcFace loss
    float s = 64.0f;
    float m = 0.50f;
    float *d_loss;
    cudaMalloc(&d_loss, batch_size * sizeof(float));
    arcface_loss_kernel<<<(batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock>>>(d_output, target, d_loss, batch_size, output_dim, s, m);

    // Copy loss to host
    cudaMemcpy(output, d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaFree(d_channel_attention_weight);
    cudaFree(d_loss);
}

}  // extern "C"
