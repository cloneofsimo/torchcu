
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for Coord Attention
__global__ void coord_attention_kernel(const float* input_tensor, float* output_tensor, int B, int C, int H, int W, int k) {
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b < B && h < H) {
        // Calculate coordinates
        float x = (float(h) / H - 0.5f) * 2.0f;
        float y = (float(b) / W - 0.5f) * 2.0f;

        // Apply kernel for each channel
        for (int c = 0; c < C; ++c) {
            // Calculate convolution output
            float conv_output = 0.0f;
            for (int kh = -k / 2; kh <= k / 2; ++kh) {
                for (int kw = -k / 2; kw <= k / 2; ++kw) {
                    int h_idx = h + kh;
                    int w_idx = b + kw;
                    if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                        conv_output += input_tensor[b * C * H * W + c * H * W + h_idx * W + w_idx];
                    }
                }
            }

            // Add coordinates to the convolution output
            conv_output += x + y;

            // Calculate attention weight
            float attention_weight = 1.0f / (1.0f + exp(-conv_output));

            // Apply attention weight
            output_tensor[b * C * H * W + c * H * W + h * W + b] = input_tensor[b * C * H * W + c * H * W + h * W + b] * attention_weight;
        }
    }
}

extern "C" {

void coord_attention_fp32(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract kernel size
    int k = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output_tensor = va_arg(args, float*);

    va_end(args);

    int B = input_tensor_dim0;
    int C = input_tensor_dim1;
    int H = input_tensor_dim2;
    int W = input_tensor_dim3;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, B * C * H * W * sizeof(float));
    cudaMalloc(&d_output, B * C * H * W * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((H + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (B + threadsPerBlock.y - 1) / threadsPerBlock.y);

    coord_attention_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, B, C, H, W, k
    );

    // Copy result back to host
    cudaMemcpy(output_tensor, d_output, B * C * H * W * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
