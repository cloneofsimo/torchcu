
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

// CUDA kernel for calculating spectral centroid
__global__ void spectral_centroid_kernel(const float* audio_data, float* centroid_data, int batch_size, int time_steps,
                                        int window_size, int hop_length, float sample_rate) {
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int time_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && time_idx < time_steps) {
        int offset = batch_idx * time_steps + time_idx;
        float sum_mag = 0.0f;
        float sum_weighted_mag = 0.0f;

        for (int freq_idx = 0; freq_idx < window_size / 2 + 1; ++freq_idx) {
            int data_idx = offset * (window_size / 2 + 1) + freq_idx;
            float mag = audio_data[data_idx];
            sum_mag += mag;
            sum_weighted_mag += mag * freq_idx;
        }

        centroid_data[offset] = (sum_weighted_mag / sum_mag) * sample_rate / window_size;
    }
}

// CUDA kernel for interpolating spectral centroid
__global__ void interpolate_kernel(const float* centroid_data, float* interpolated_data, int batch_size, int time_steps,
                                   int target_time_steps) {
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int time_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && time_idx < target_time_steps) {
        float ratio = (float)time_idx / (target_time_steps - 1);
        int left_idx = (int)(ratio * (time_steps - 1));
        int right_idx = (int)ceilf(ratio * (time_steps - 1));

        float weight = ratio * (time_steps - 1) - left_idx;

        interpolated_data[batch_idx * target_time_steps + time_idx] = 
            (1 - weight) * centroid_data[batch_idx * time_steps + left_idx] +
            weight * centroid_data[batch_idx * time_steps + right_idx];
    }
}

// CUDA kernel for group normalization
__global__ void group_norm_kernel(const float* input_data, float* output_data, int batch_size, int time_steps,
                                  int num_groups) {
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int time_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && time_idx < time_steps) {
        int group_idx = (batch_idx * time_steps + time_idx) / num_groups;
        int group_offset = group_idx * num_groups;

        float sum = 0.0f;
        for (int i = 0; i < num_groups; ++i) {
            sum += input_data[group_offset + i];
        }
        float mean = sum / num_groups;

        float sum_sq = 0.0f;
        for (int i = 0; i < num_groups; ++i) {
            sum_sq += (input_data[group_offset + i] - mean) * (input_data[group_offset + i] - mean);
        }
        float variance = sum_sq / num_groups;

        output_data[batch_idx * time_steps + time_idx] = (input_data[batch_idx * time_steps + time_idx] - mean) / sqrtf(variance + 1e-5);
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input data
    const float* audio_data = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int time_steps = va_arg(args, int);

    // Extract other arguments
    int sample_rate = va_arg(args, int);
    int window_size = va_arg(args, int);
    int hop_length = va_arg(args, int);

    // Extract output tensor
    float* output_data = va_arg(args, float*);

    va_end(args);

    // Allocate device memory for intermediate data
    float* d_centroid_data;
    float* d_interpolated_data;
    cudaMalloc(&d_centroid_data, batch_size * time_steps * sizeof(float));
    cudaMalloc(&d_interpolated_data, batch_size * time_steps * sizeof(float));

    // Launch kernels
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((time_steps + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    spectral_centroid_kernel<<<numBlocks, threadsPerBlock>>>(audio_data, d_centroid_data, batch_size, time_steps, window_size, hop_length, sample_rate);

    interpolate_kernel<<<numBlocks, threadsPerBlock>>>(d_centroid_data, d_interpolated_data, batch_size, time_steps, time_steps);

    group_norm_kernel<<<numBlocks, threadsPerBlock>>>(d_interpolated_data, output_data, batch_size, time_steps, 4);

    // Free device memory
    cudaFree(d_centroid_data);
    cudaFree(d_interpolated_data);
}

}
