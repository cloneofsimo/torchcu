
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

// Helper function for GELU activation (approximation using tanh)
__device__ __forceinline__ float gelu_approx(float x) {
    const float c = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.7978845608028654f * x * x * x)));
}

// CUDA kernel for spatial attention, GELU, and layer scaling decay
__global__ void spatial_attention_gelu_fp16_kernel(
    const half* input, const half* weight, float* output, float scaling_factor,
    int batch_size, int input_dim, int height, int width, int attention_dim) {

    int b = blockIdx.z * blockDim.z + threadIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && h < height && w < width) {
        float sum = 0.0f;
        for (int k = 0; k < attention_dim; ++k) {
            float exp_sum = 0.0f;
            for (int j = 0; j < input_dim; ++j) {
                float val = __int2float_rn(input[(b * height + h) * width * input_dim + w * input_dim + j]);
                float w_val = __int2float_rn(weight[k * input_dim + j]);
                exp_sum += val * w_val;
            }
            float attention = exp(exp_sum); // Softmax via exp and normalization
            for (int j = 0; j < input_dim; ++j) {
                float val = __int2float_rn(input[(b * height + h) * width * input_dim + w * input_dim + j]);
                float w_val = __int2float_rn(weight[k * input_dim + j]);
                sum += attention * val * w_val;
            }
        }
        output[(b * height + h) * width * input_dim + w * input_dim] = gelu_approx(sum) * scaling_factor;
    }
}

extern "C" {
void spatial_attention_gelu_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const half* input = va_arg(args, const half*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);
    int input_dim2 = va_arg(args, int);
    int input_dim3 = va_arg(args, int);

    // Extract weight tensor
    const half* weight = va_arg(args, const half*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract scaling factor
    float scaling_factor = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_dim0;
    int input_dim = input_dim1;
    int height = input_dim2;
    int width = input_dim3;
    int attention_dim = weight_dim0;

    // Allocate device memory
    half *d_input, *d_weight;
    float *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * height * width * sizeof(half));
    cudaMalloc(&d_weight, attention_dim * input_dim * sizeof(half));
    cudaMalloc(&d_output, batch_size * input_dim * height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * input_dim * height * width * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, attention_dim * input_dim * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    spatial_attention_gelu_fp16_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, scaling_factor,
        batch_size, input_dim, height, width, attention_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_dim * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}
}  // extern "C"
