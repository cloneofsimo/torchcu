
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <stdarg.h> 

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for CoordAttention with bfloat16
__global__ void coord_attention_kernel_bf16(const float* input_tensor, const float* weight, float* output, 
                                        int B, int C, int H, int W, int kernel_size, 
                                        int padding) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < B && h < H && w < W) {
        // Coordinate embedding
        int x = w;
        int y = h;
        float x_embed = (float)x / (W - 1);
        float y_embed = (float)y / (H - 1);

        // Convolution using bfloat16
        __nv_bfloat16 sum = 0;
        for (int k = 0; k < kernel_size; ++k) {
            for (int l = 0; l < kernel_size; ++l) {
                int i = h + k - padding;
                int j = w + l - padding;
                if (i >= 0 && i < H && j >= 0 && j < W) {
                    for (int c = 0; c < C; ++c) {
                        __nv_bfloat16 input_val = float_to_bfloat16(input_tensor[b * C * H * W + c * H * W + i * W + j]);
                        __nv_bfloat16 weight_val = float_to_bfloat16(weight[(k * kernel_size + l) * C + c]);
                        sum += __hmul(input_val, weight_val);
                    }
                }
            }
        }
        float conv_output = bfloat16_to_float(sum);

        // Cosine similarity
        __nv_bfloat16 input_val = float_to_bfloat16(input_tensor[b * C * H * W + c * H * W + h * W + w]);
        float cosine_sim = __fdividef(conv_output, sqrtf(input_val * conv_output));

        // Attention and output
        output[b * C * H * W + c * H * W + h * W + w] = cosine_sim * input_tensor[b * C * H * W + c * H * W + h * W + w];
    }
}

extern "C" {

void coord_attention_bf16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int B = va_arg(args, int);
    int C = va_arg(args, int);
    int H = va_arg(args, int);
    int W = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int kernel_size = va_arg(args, int);
    int padding = va_arg(args, int);

    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, B * C * H * W * sizeof(float));
    cudaMalloc(&d_weight, kernel_size * kernel_size * C * sizeof(float));
    cudaMalloc(&d_output, B * C * H * W * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, kernel_size * kernel_size * C * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((B + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (H + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (W + threadsPerBlock.z - 1) / threadsPerBlock.z);

    coord_attention_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, B, C, H, W, kernel_size, padding
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, B * C * H * W * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
