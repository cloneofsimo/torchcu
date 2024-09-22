
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define WARPS_PER_BLOCK 8

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// Kernel for low-rank approximation, element-wise product, fused layer norm, max pooling, and conversion to fp16
__global__ void low_rank_approximation_int8_kernel(const int8_t* input, const int8_t* weight, const int8_t* bias,
                                                half* output, int batch, int input_dim, int low_rank_dim, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * output_dim) {
        int b = idx / output_dim;
        int o = idx % output_dim;

        float sum = 0.0f;
        for (int i = 0; i < low_rank_dim; ++i) {
            sum += (float)input[b * input_dim + i] * (float)weight[o * low_rank_dim + i];
        }
        sum *= (float)bias[o];

        // Fused Layer Normalization
        // Calculate mean and variance (using shared memory for efficiency)
        __shared__ float shared_sum[WARPS_PER_BLOCK];
        __shared__ float shared_sq_sum[WARPS_PER_BLOCK];
        if (threadIdx.x < WARPS_PER_BLOCK) {
            shared_sum[threadIdx.x] = 0.0f;
            shared_sq_sum[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        float local_sum = 0.0f;
        float local_sq_sum = 0.0f;
        for (int i = 0; i < low_rank_dim; ++i) {
            local_sum += (float)input[b * input_dim + i] * (float)weight[o * low_rank_dim + i];
        }
        local_sq_sum = local_sum * local_sum;

        atomicAdd(&shared_sum[threadIdx.x / 32], local_sum);
        atomicAdd(&shared_sq_sum[threadIdx.x / 32], local_sq_sum);
        __syncthreads();

        if (threadIdx.x == 0) {
            float warp_sum = 0.0f;
            float warp_sq_sum = 0.0f;
            for (int i = 0; i < WARPS_PER_BLOCK; ++i) {
                warp_sum += shared_sum[i];
                warp_sq_sum += shared_sq_sum[i];
            }
            float mean = warp_sum / (float)low_rank_dim;
            float variance = (warp_sq_sum / (float)low_rank_dim) - (mean * mean);
            variance = fmaxf(variance, 1e-5f); // Avoid numerical issues
            sum = (sum - mean) / sqrtf(variance);
        }
        __syncthreads();

        // Max pooling
        int pool_idx = (idx / output_dim) * (output_dim / 2) + (idx % output_dim) / 2;
        if (idx % 2 == 0 && pool_idx < batch * (output_dim / 2)) {
            output[pool_idx] = fmaxf(float_to_half(sum), output[pool_idx]);
        }
    }
}

extern "C" {

void low_rank_approximation_int8_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const int8_t* input = va_arg(args, const int8_t*);
    int batch = va_arg(args, int);
    int input_dim = va_arg(args, int);
    const int8_t* weight = va_arg(args, const int8_t*);
    int low_rank_dim = va_arg(args, int);
    int output_dim = va_arg(args, int);
    const int8_t* bias = va_arg(args, const int8_t*);

    // Extract output tensor
    half* output = va_arg(args, half*);

    va_end(args);

    // Allocate device memory
    int8_t *d_input, *d_weight, *d_bias;
    half *d_output;
    cudaMalloc(&d_input, batch * input_dim * sizeof(int8_t));
    cudaMalloc(&d_weight, output_dim * low_rank_dim * sizeof(int8_t));
    cudaMalloc(&d_bias, output_dim * sizeof(int8_t));
    cudaMalloc(&d_output, batch * (output_dim / 2) * sizeof(half));

    // Copy data to device
    cudaMemcpy(d_input, input, batch * input_dim * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * low_rank_dim * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_dim * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int num_blocks = (batch * output_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    low_rank_approximation_int8_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_input, d_weight, d_bias, d_output, batch, input_dim, low_rank_dim, output_dim);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch * (output_dim / 2) * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
