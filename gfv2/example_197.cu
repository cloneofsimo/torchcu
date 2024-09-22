
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

// CUDA kernel for matrix multiplication
__global__ void bmm_kernel_bf16(const float* input, const float* weight, float* output, 
                                int batch_size, int input_dim, int output_dim) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b < batch_size && i < input_dim) {
        float sum = 0.0f;
        for (int j = 0; j < output_dim; j++) {
            __nv_bfloat16 a = float_to_bfloat16(input[b * input_dim + i]);
            __nv_bfloat16 b = float_to_bfloat16(weight[j * input_dim + i]);
            sum += bfloat16_to_float(__hmul(a, b));
        }
        output[b * output_dim + j] = sum;
    }
}

__global__ void window_attention_kernel(const float* input, float* output, 
                                       const bool* mask, int batch_size, int h, int w, int c, int window_size, int num_heads) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = threadIdx.z;

    if (b < batch_size && y < h && x < w) {
        int window_index_y = y / window_size;
        int window_index_x = x / window_size;
        int window_offset_y = y % window_size;
        int window_offset_x = x % window_size;
        int window_index = window_index_y * window_size + window_index_x;
        int window_offset = window_offset_y * window_size + window_offset_x;

        int num_heads_per_window = num_heads / (window_size * window_size);

        // Calculate attention scores
        __shared__ float attn_scores[32 * 32];  // Assuming max window size of 8x8
        float sum = 0.0f;
        for (int head = 0; head < num_heads_per_window; ++head) {
            for (int j = 0; j < window_size * window_size; ++j) {
                int mask_index = b * window_size * window_size + window_index * window_size * window_size + j;
                if (mask[mask_index]) {
                    int input_index_1 = b * h * w * c + (y * w + x) * c + head * c / num_heads;
                    int input_index_2 = b * h * w * c + (window_index * window_size * window_size + j) * c + head * c / num_heads;
                    float score = 0.0f;
                    for (int k = 0; k < c / num_heads; ++k) {
                        score += input[input_index_1 + k] * input[input_index_2 + k];
                    }
                    attn_scores[(head * window_size * window_size + j) * window_size * window_size + window_offset] = score;
                } else {
                    attn_scores[(head * window_size * window_size + j) * window_size * window_size + window_offset] = -INFINITY;
                }
            }
        }
        __syncthreads();

        // Softmax
        for (int head = 0; head < num_heads_per_window; ++head) {
            for (int j = 0; j < window_size * window_size; ++j) {
                attn_scores[(head * window_size * window_size + j) * window_size * window_size + window_offset] = expf(attn_scores[(head * window_size * window_size + j) * window_size * window_size + window_offset]);
            }
        }
        __syncthreads();

        float sum_exp = 0.0f;
        for (int j = 0; j < window_size * window_size; ++j) {
            sum_exp += attn_scores[(head * window_size * window_size + j) * window_size * window_size + window_offset];
        }
        __syncthreads();

        for (int j = 0; j < window_size * window_size; ++j) {
            attn_scores[(head * window_size * window_size + j) * window_size * window_size + window_offset] /= sum_exp;
        }
        __syncthreads();

        // Weighted sum
        for (int head = 0; head < num_heads_per_window; ++head) {
            for (int j = 0; j < window_size * window_size; ++j) {
                int input_index = b * h * w * c + (window_index * window_size * window_size + j) * c + head * c / num_heads;
                float value = 0.0f;
                for (int k = 0; k < c / num_heads; ++k) {
                    value += input[input_index + k] * attn_scores[(head * window_size * window_size + j) * window_size * window_size + window_offset];
                }
                output[b * h * w * c + (y * w + x) * c + head * c / num_heads + k] = value;
            }
        }
    }
}

extern "C" {

void attention_with_quantization_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract mask tensor (optional)
    const bool* mask = NULL;
    int mask_dim0 = 0;
    int mask_dim1 = 0;
    int mask_dim2 = 0;
    int mask_dim3 = 0;
    if (num_args > 6) {
        mask = va_arg(args, const bool*);
        mask_dim0 = va_arg(args, int);
        mask_dim1 = va_arg(args, int);
        mask_dim2 = va_arg(args, int);
        mask_dim3 = va_arg(args, int);
    }

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    bool *d_mask = NULL;
    cudaMalloc(&d_input, batch_size * input_dim * input_tensor_dim2 * sizeof(float));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * input_tensor_dim2 * sizeof(float));
    if (mask != NULL) {
        cudaMalloc(&d_mask, mask_dim0 * mask_dim1 * mask_dim2 * mask_dim3 * sizeof(bool));
    }

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * input_tensor_dim2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    if (mask != NULL) {
        cudaMemcpy(d_mask, mask, mask_dim0 * mask_dim1 * mask_dim2 * mask_dim3 * sizeof(bool), cudaMemcpyHostToDevice);
    }

    // Launch window attention kernel
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_tensor_dim1 + threadsPerBlock.y - 1) / threadsPerBlock.y);
    window_attention_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, d_mask, batch_size, input_tensor_dim1, input_tensor_dim2, input_tensor_dim2, 8, 8
    );

    // Launch bmm kernel
    threadsPerBlock = dim3(32, 32);
    numBlocks = dim3((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (input_tensor_dim1 + threadsPerBlock.y - 1) / threadsPerBlock.y);
    bmm_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_output, d_weight, d_output, batch_size, input_tensor_dim1, output_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * input_tensor_dim2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    if (mask != NULL) {
        cudaFree(d_mask);
    }
}

}  // extern "C"
