
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

#define THREADS_PER_BLOCK 256
#define WARPS_PER_BLOCK 8

__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

__device__ __forceinline__ float exp_bf16(float x) {
  __nv_bfloat16 x_bf16 = float_to_bfloat16(x);
  return bfloat16_to_float(__expf(x_bf16));
}

// Shared memory for warp-level reduction
__shared__ float sdata[THREADS_PER_BLOCK];

template <typename T>
__device__ void warpReduceSum(T &val) {
  for (int i = WARPS_PER_BLOCK / 2; i > 0; i >>= 1) {
    val += __shfl_down(val, i);
  }
}

__global__ void my_function_kernel(const float* input, const int* labels, float* output, 
                                    int batch_size, int input_features, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        // Linear layer 1
        float sum1 = 0.0f;
        for (int i = 0; i < input_features; ++i) {
            sum1 += input[idx * input_features + i];
        }
        __nv_bfloat16 sum1_bf16 = float_to_bfloat16(sum1);

        // ReLU activation
        sum1_bf16 = __fmaxf(sum1_bf16, float_to_bfloat16(0.0f));

        // Batch Normalization
        // (Simplified implementation for demonstration, no running mean/variance)
        float mean = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            mean += input[i * input_features + idx];
        }
        mean /= batch_size;
        
        float var = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            var += (input[i * input_features + idx] - mean) * (input[i * input_features + idx] - mean);
        }
        var /= batch_size;
        
        // Assuming gamma = 1 and beta = 0 for simplicity
        float bn_output = (input[idx * input_features + idx] - mean) / sqrt(var + 1e-5);

        // Linear layer 2
        __nv_bfloat16 sum2_bf16 = float_to_bfloat16(0.0f);
        for (int i = 0; i < 128; ++i) {
            sum2_bf16 += float_to_bfloat16(bn_output) * float_to_bfloat16(0.5f);  // Example weights
        }

        // LogSoftmax
        float max_val = -INFINITY;
        for (int i = 0; i < num_classes; ++i) {
            max_val = fmaxf(max_val, bfloat16_to_float(sum2_bf16 * float_to_bfloat16(i)));
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            sum_exp += exp_bf16(bfloat16_to_float(sum2_bf16 * float_to_bfloat16(i)) - max_val);
        }

        for (int i = 0; i < num_classes; ++i) {
            output[idx * num_classes + i] = logf(exp_bf16(bfloat16_to_float(sum2_bf16 * float_to_bfloat16(i)) - max_val) / sum_exp);
        }
    }
}

extern "C" {
    void my_function(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract input tensor
        const float* input = va_arg(args, const float*);
        int batch_size = va_arg(args, int);
        int input_features = va_arg(args, int);

        // Extract labels tensor
        const int* labels = va_arg(args, const int*);

        // Extract output tensor
        float* output = va_arg(args, float*);

        int num_classes = labels[batch_size - 1] + 1;

        va_end(args);

        // Allocate device memory
        float *d_input, *d_output;
        cudaMalloc(&d_input, batch_size * input_features * sizeof(float));
        cudaMalloc(&d_output, batch_size * num_classes * sizeof(float));

        // Copy input data to device
        cudaMemcpy(d_input, input, batch_size * input_features * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        int num_blocks = (batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        my_function_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_input, labels, d_output, batch_size, input_features, num_classes
        );

        // Copy result back to host
        cudaMemcpy(output, d_output, batch_size * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
    }
}
