
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

extern "C" {

// CUDA kernel for noise injection, norm calculation, and any check
__global__ void noisy_norm_any_kernel(const float* input_tensor, const float* noise_scale, bool* output, 
                                        int batch_size, int feature_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < feature_dim; ++i) {
            float val = input_tensor[idx * feature_dim + i] + *noise_scale * curand_uniform();
            sum += val * val;
        }
        output[idx] = (sqrtf(sum) > 1.0f);
    }
}

// CUDA kernel for noise injection, norm calculation, and any check
__global__ void noisy_norm_any_kernel_fp16(const half* input_tensor, const half* noise_scale, bool* output, 
                                        int batch_size, int feature_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < feature_dim; ++i) {
            float val = __half2float(input_tensor[idx * feature_dim + i]) + __half2float(*noise_scale) * curand_uniform();
            sum += val * val;
        }
        output[idx] = (sqrtf(sum) > 1.0f);
    }
}

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract noise scale
    const float* noise_scale = va_arg(args, const float*);

    // Extract output tensor (assuming it's preallocated)
    bool* output = va_arg(args, bool*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int feature_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_noise_scale;
    bool *d_output;
    cudaMalloc(&d_input, batch_size * feature_dim * sizeof(float));
    cudaMalloc(&d_noise_scale, sizeof(float));
    cudaMalloc(&d_output, batch_size * sizeof(bool));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_noise_scale, noise_scale, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(1024);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    noisy_norm_any_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_noise_scale, d_output, batch_size, feature_dim);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(bool), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_noise_scale);
    cudaFree(d_output);
}

void torch_function_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const half* input_tensor = va_arg(args, const half*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract noise scale
    const half* noise_scale = va_arg(args, const half*);

    // Extract output tensor (assuming it's preallocated)
    bool* output = va_arg(args, bool*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int feature_dim = input_tensor_dim1;

    // Allocate device memory
    half *d_input, *d_noise_scale;
    bool *d_output;
    cudaMalloc(&d_input, batch_size * feature_dim * sizeof(half));
    cudaMalloc(&d_noise_scale, sizeof(half));
    cudaMalloc(&d_output, batch_size * sizeof(bool));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * feature_dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_noise_scale, noise_scale, sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(1024);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    noisy_norm_any_kernel_fp16<<<numBlocks, threadsPerBlock>>>(d_input, d_noise_scale, d_output, batch_size, feature_dim);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(bool), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_noise_scale);
    cudaFree(d_output);
}

}  // extern "C"
