
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>

// Swiglu activation (C++ version)
__device__ __forceinline__ float swiglu(float x) {
    return x * (x > 0.0f);
}

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

__global__ void image_jacobian_kernel(const float* input_tensor, const float* weight, const float* bias, float* output,
                                     int batch_size, int input_channels, int input_height, int input_width,
                                     int output_size) {
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && output_idx < output_size) {
        float sum = 0.0f;
        for (int c = 0; c < input_channels; ++c) {
            for (int h = 0; h < input_height; ++h) {
                for (int w = 0; w < input_width; ++w) {
                    int input_offset = batch_idx * input_channels * input_height * input_width + 
                                       c * input_height * input_width + h * input_width + w;
                    int weight_offset = output_idx * input_channels * input_height * input_width + 
                                       c * input_height * input_width + h * input_width + w;

                    __nv_bfloat16 input_bf16 = float_to_bfloat16(input_tensor[input_offset]);
                    __nv_bfloat16 weight_bf16 = float_to_bfloat16(weight[weight_offset]);
                    sum += bfloat16_to_float(__hmul(input_bf16, weight_bf16));
                }
            }
        }

        sum += bias[output_idx];
        output[batch_idx * output_size + output_idx] = swiglu(sum);
    }
}


__global__ void image_jacobian_kernel_derivative(const float* input_tensor, const float* weight, const float* bias, float* output,
                                        int batch_size, int input_channels, int input_height, int input_width,
                                        int output_size) {
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && input_idx < input_channels * input_height * input_width) {
        int output_offset = batch_idx * output_size;

        // Swiglu derivative: 1 if input > 0, 0 otherwise
        float derivative = input_tensor[batch_idx * input_channels * input_height * input_width + input_idx] > 0.0f ? 1.0f : 0.0f;

        // Compute the Jacobian entry for the current input element
        for (int output_idx = 0; output_idx < output_size; ++output_idx) {
            int weight_offset = output_idx * input_channels * input_height * input_width + input_idx;
            output[output_offset + output_idx] = derivative * weight[weight_offset];
        }
    }
}

extern "C" {

void image_jacobian_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_channels = input_tensor_dim1;
    int input_height = input_tensor_dim2;
    int input_width = input_tensor_dim3;
    int output_size = weight_dim0;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * input_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * input_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_bias, bias_dim0 * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * input_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    image_jacobian_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output, batch_size, input_channels, input_height, input_width, output_size
    );

    // Launch kernel for Jacobian derivative computation
    dim3 threadsPerBlock2(32, 32);
    dim3 numBlocks2((input_channels * input_height * input_width + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
                   (batch_size + threadsPerBlock2.y - 1) / threadsPerBlock2.y);
    
    image_jacobian_kernel_derivative<<<numBlocks2, threadsPerBlock2>>>(
        d_input, d_weight, d_bias, d_output, batch_size, input_channels, input_height, input_width, output_size
    );
    
    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

} // extern "C"
