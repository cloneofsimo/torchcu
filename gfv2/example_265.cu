
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for L1 loss calculation
__global__ void l1_loss_kernel(const float* input, const float* weight, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fabsf(input[idx] - weight[idx]);
    }
}

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* eye_matrix, const float* input, float* output, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += eye_matrix[row * k + i] * input[i * n + col];
        }
        output[row * n + col] = sum;
    }
}

// CUDA kernel for elementwise division
__global__ void elementwise_div_kernel(const float* input, const float* weight, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / weight[idx];
    }
}

// CUDA kernel for ReLU activation
__global__ void relu_kernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = fmaxf(input[idx], 0.0f);
    }
}

// CUDA kernel for in-place addition
__global__ void inplace_add_kernel(float* input, const float* bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] += bias[idx];
    }
}

extern "C" {

void my_custom_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    const float* bias = va_arg(args, const float*);
    int bias_dim = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    half* output = va_arg(args, half*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_l1_loss, *d_eye_matrix, *d_identity_result, *d_elementwise_div_result, *d_activation_result;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * sizeof(float));
    cudaMalloc(&d_bias, bias_dim * sizeof(float));
    cudaMalloc(&d_l1_loss, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_eye_matrix, batch_size * batch_size * sizeof(float));
    cudaMalloc(&d_identity_result, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_elementwise_div_result, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_activation_result, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_dim * sizeof(float), cudaMemcpyHostToDevice);

    // L1 Loss calculation
    l1_loss_kernel<<<(batch_size * input_dim + 255) / 256, 256>>>(d_input, d_weight, d_l1_loss, batch_size * input_dim);

    // Identity matrix multiplication
    for (int i = 0; i < batch_size; ++i) {
        d_eye_matrix[i * batch_size + i] = 1.0f;
    }
    matmul_kernel<<<(input_dim + 15) / 16, 16>>>(d_eye_matrix, d_input, d_identity_result, batch_size, input_dim, batch_size);

    // Elementwise division
    elementwise_div_kernel<<<(batch_size * input_dim + 255) / 256, 256>>>(d_identity_result, d_weight, d_elementwise_div_result, batch_size * input_dim);

    // ReLU activation
    relu_kernel<<<(batch_size * input_dim + 255) / 256, 256>>>(d_elementwise_div_result, batch_size * input_dim);

    // In-place addition
    inplace_add_kernel<<<(batch_size * input_dim + 255) / 256, 256>>>(d_elementwise_div_result, d_bias, batch_size * input_dim);

    // Copy result back to host (in fp16)
    cudaMemcpy(output, d_elementwise_div_result, batch_size * input_dim * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_l1_loss);
    cudaFree(d_eye_matrix);
    cudaFree(d_identity_result);
    cudaFree(d_elementwise_div_result);
    cudaFree(d_activation_result);
}

}  // extern "C"
