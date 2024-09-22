
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

__global__ void conv_bf16_kernel(const float* input, const float* weight, const float* bias, float* output,
                                 int batch_size, int input_channels, int input_size, int output_channels,
                                 int kernel_size, int stride, int padding) {

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_x < input_size - kernel_size + 2 * padding + 1 &&
        out_y < batch_size &&
        out_c < output_channels) {

        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            for (int ic = 0; ic < input_channels; ic++) {
                int in_x = out_x + k - padding;
                int in_y = out_y;
                int in_c = ic;

                if (in_x >= 0 && in_x < input_size) {
                    int in_idx = in_y * input_size * input_channels + in_x * input_channels + in_c;
                    int w_idx = out_c * kernel_size * input_channels + k * input_channels + ic;
                    __nv_bfloat16 a = float_to_bfloat16(input[in_idx]);
                    __nv_bfloat16 b = float_to_bfloat16(weight[w_idx]);
                    sum += bfloat16_to_float(__hmul(a, b));
                }
            }
        }

        sum += bias[out_c];
        output[out_y * input_size * output_channels + out_x * output_channels + out_c] = fmaxf(sum, 0.0f);
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);
    int input_dim2 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);

    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    int stride = va_arg(args, int);
    int padding = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_dim0;
    int input_channels = input_dim1;
    int input_size = input_dim2;
    int output_channels = weight_dim0;
    int kernel_size = weight_dim2;

    // Calculate output size
    int output_size = (input_size + 2 * padding - kernel_size) / stride + 1;

    // Allocate device memory
    float* d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * input_size * input_channels * sizeof(float));
    cudaMalloc(&d_weight, output_channels * kernel_size * input_channels * sizeof(float));
    cudaMalloc(&d_bias, output_channels * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_size * output_channels * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input, batch_size * input_size * input_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_channels * kernel_size * input_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((output_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (output_channels + threadsPerBlock.z - 1) / threadsPerBlock.z);

    conv_bf16_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_bias, d_output, 
                                                     batch_size, input_channels, input_size, output_channels,
                                                     kernel_size, stride, padding);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_size * output_channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

} // extern "C"
