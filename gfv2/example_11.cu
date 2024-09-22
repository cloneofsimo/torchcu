
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/epilogue/threadblock/epilogue.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/util/tensor_view.h>
#include <cutlass/transform/threadblock/transform.h>
#include <cutlass/epilogue/threadblock/fast_int8.h>
#include <cutlass/epilogue/threadblock/fast_fp16.h>
#include <cutlass/epilogue/threadblock/linear_combination.h>

#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

#define kThreadblockRows 128
#define kThreadblockCols 16
#define kWarpSize 32
#define kSMEM_BYTES (kThreadblockRows * kThreadblockCols * sizeof(int8_t))

using namespace cutlass;

// CUDA kernel for matrix multiplication and ReLU using bfloat16
__global__ void token_mixing_function_kernel(
    const int8_t* input_tensor,
    const int8_t* weight_qkv,
    const int8_t* weight_out,
    const int8_t* norm_weight,
    const int8_t* norm_bias,
    float* output,
    int batch_size,
    int seq_len,
    int d_model
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < seq_len) {
        int8_t* smem = reinterpret_cast<int8_t*>(__ldg(&shared_mem[0]));

        // LayerNorm
        int8_t* norm_input = reinterpret_cast<int8_t*>(input_tensor + (row * seq_len + col) * d_model);
        float norm_sum = 0.0f;
        for (int i = 0; i < d_model; i++) {
            norm_sum += norm_input[i] * norm_weight[i];
        }
        float norm_result = (norm_sum + norm_bias[0]) / sqrtf(d_model);

        // Multiply by qkv weight
        for (int i = 0; i < d_model; i++) {
            smem[threadIdx.y * d_model + i] = norm_input[i] * weight_qkv[col * d_model * 3 + i];
        }
        __syncthreads();

        // Attention
        float q[kThreadblockCols] = {0};
        float k[kThreadblockCols] = {0};
        float v[kThreadblockCols] = {0};

        for (int i = 0; i < kThreadblockCols; i++) {
            q[i] = smem[threadIdx.y * d_model + i];
            k[i] = smem[threadIdx.y * d_model + i + d_model];
            v[i] = smem[threadIdx.y * d_model + i + 2 * d_model];
        }

        float attn[kThreadblockCols] = {0};
        for (int i = 0; i < kThreadblockCols; i++) {
            for (int j = 0; j < kThreadblockCols; j++) {
                attn[i] += q[i] * k[j] / sqrtf(d_model);
            }
        }

        // Softmax
        float sum = 0.0f;
        for (int i = 0; i < kThreadblockCols; i++) {
            sum += expf(attn[i]);
        }
        for (int i = 0; i < kThreadblockCols; i++) {
            attn[i] = expf(attn[i]) / sum;
        }

        // Apply attention
        float result[kThreadblockCols] = {0};
        for (int i = 0; i < kThreadblockCols; i++) {
            for (int j = 0; j < kThreadblockCols; j++) {
                result[i] += attn[i] * v[j];
            }
        }

        // Multiply by weight_out
        float output_sum = 0.0f;
        for (int i = 0; i < kThreadblockCols; i++) {
            output_sum += result[i] * weight_out[i * d_model + threadIdx.x];
        }
        output[(row * seq_len + col) * d_model + threadIdx.x] = output_sum;
    }
}

extern "C" {
void token_mixing_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const int8_t* input_tensor = va_arg(args, const int8_t*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract weight_qkv tensor
    const int8_t* weight_qkv = va_arg(args, const int8_t*);
    int weight_qkv_dim0 = va_arg(args, int);
    int weight_qkv_dim1 = va_arg(args, int);

    // Extract weight_out tensor
    const int8_t* weight_out = va_arg(args, const int8_t*);
    int weight_out_dim0 = va_arg(args, int);
    int weight_out_dim1 = va_arg(args, int);

    // Extract norm_weight tensor
    const int8_t* norm_weight = va_arg(args, const int8_t*);
    int norm_weight_dim0 = va_arg(args, int);

    // Extract norm_bias tensor
    const int8_t* norm_bias = va_arg(args, const int8_t*);
    int norm_bias_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Calculate kernel dimensions
    int batch_size = input_tensor_dim0;
    int seq_len = input_tensor_dim1;
    int d_model = input_tensor_dim2;

    // Allocate device memory
    int8_t *d_input, *d_weight_qkv, *d_weight_out, *d_norm_weight, *d_norm_bias;
    float *d_output;
    cudaMalloc(&d_input, batch_size * seq_len * d_model * sizeof(int8_t));
    cudaMalloc(&d_weight_qkv, weight_qkv_dim0 * weight_qkv_dim1 * sizeof(int8_t));
    cudaMalloc(&d_weight_out, weight_out_dim0 * weight_out_dim1 * sizeof(int8_t));
    cudaMalloc(&d_norm_weight, norm_weight_dim0 * sizeof(int8_t));
    cudaMalloc(&d_norm_bias, norm_bias_dim0 * sizeof(int8_t));
    cudaMalloc(&d_output, batch_size * seq_len * d_model * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * seq_len * d_model * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_qkv, weight_qkv, weight_qkv_dim0 * weight_qkv_dim1 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_out, weight_out, weight_out_dim0 * weight_out_dim1 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm_weight, norm_weight, norm_weight_dim0 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm_bias, norm_bias, norm_bias_dim0 * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(kThreadblockCols, kThreadblockRows);
    dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    token_mixing_function_kernel<<<numBlocks, threadsPerBlock, kSMEM_BYTES>>>(
        d_input, d_weight_qkv, d_weight_out, d_norm_weight, d_norm_bias, d_output,
        batch_size, seq_len, d_model
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight_qkv);
    cudaFree(d_weight_out);
    cudaFree(d_norm_weight);
    cudaFree(d_norm_bias);
    cudaFree(d_output);
}

} // extern "C"
