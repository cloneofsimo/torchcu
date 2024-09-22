
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for Instance Normalization with FP16
__global__ void instance_norm_fp16_kernel(const half* input_tensor, const half* gamma, const half* beta, 
                                         half* output, int batch_size, int channels, int height, int width) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch_size && channel_idx < channels) {
        float sum = 0.0f;
        float sq_sum = 0.0f;

        // Calculate mean and variance for the current channel and instance
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float val = __half2float(input_tensor[(batch_idx * channels + channel_idx) * height * width + h * width + w]);
                sum += val;
                sq_sum += val * val;
            }
        }

        float mean = sum / (height * width);
        float variance = sq_sum / (height * width) - mean * mean;

        // Normalize the input
        float std = sqrtf(variance + 1e-5f);  // Add small constant for numerical stability
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float val = __half2float(input_tensor[(batch_idx * channels + channel_idx) * height * width + h * width + w]);
                float normalized_val = (val - mean) / std;

                // Apply gamma and beta
                output[(batch_idx * channels + channel_idx) * height * width + h * width + w] = 
                    __float2half(normalized_val * __half2float(gamma[channel_idx]) + __half2float(beta[channel_idx]));
            }
        }
    }
}

extern "C" {

void instance_norm_fp16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const half* input_tensor = va_arg(args, const half*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract gamma tensor
    const half* gamma = va_arg(args, const half*);
    int gamma_dim0 = va_arg(args, int);

    // Extract beta tensor
    const half* beta = va_arg(args, const half*);
    int beta_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    half* output = va_arg(args, half*);

    va_end(args);

    // Launch the kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_tensor_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_tensor_dim1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    instance_norm_fp16_kernel<<<numBlocks, threadsPerBlock>>>(
        input_tensor, gamma, beta, output, input_tensor_dim0, input_tensor_dim1, input_tensor_dim2, input_tensor_dim3
    );
}

}  // extern "C"
