
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>
#include <math.h>
#include <cutlass/cutlass.h>

using namespace cutlass;

// Helper function to convert float to __nv_fp16
__device__ __forceinline__ __nv_fp16 float_to_fp16(float f) {
    return __float2half_rn(f);
}

// Helper function to convert __nv_fp16 to float
__device__ __forceinline__ float fp16_to_float(__nv_fp16 h) {
    return __half2float(h);
}

// CUDA kernel for softplus
__global__ void softplus_kernel_fp16(const float* input, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        __nv_fp16 val = float_to_fp16(input[i]);
        output[i] = fp16_to_float(logf(expf(fp16_to_float(val)) + 1.0f)); // Softplus using fp16
    }
}

// CUDA kernel for time stretch (linear interpolation)
__global__ void time_stretch_kernel_fp16(const float* input, float* output, int batch_size, int channels,
                                            int input_time, int output_time, float time_stretch_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < batch_size && j < channels && k < output_time) {
        // Calculate the input time index based on the output time index
        float input_index = (float)k / time_stretch_factor; 
        int left_index = floorf(input_index);
        int right_index = ceilf(input_index);

        // Clamp indices to avoid out-of-bounds access
        left_index = min(max(0, left_index), input_time - 1);
        right_index = min(max(0, right_index), input_time - 1);

        // Perform linear interpolation
        float alpha = input_index - (float)left_index;
        output[i * channels * output_time + j * output_time + k] = 
            (1 - alpha) * input[i * channels * input_time + j * input_time + left_index] + 
            alpha * input[i * channels * input_time + j * input_time + right_index];
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int channels = va_arg(args, int);
    int input_time = va_arg(args, int);
    int input_dim = va_arg(args, int);

    // Extract time stretch factor
    float time_stretch_factor = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int output_time = int(input_time * time_stretch_factor);

    // Allocate device memory
    float *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, batch_size * channels * input_time * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * output_time * sizeof(float));
    cudaMalloc(&d_temp, batch_size * channels * input_time * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * channels * input_time * sizeof(float), cudaMemcpyHostToDevice);

    // Softplus
    softplus_kernel_fp16<<<(batch_size * channels * input_time + 255) / 256, 256>>>(d_input, d_temp, batch_size * channels * input_time);

    // Time Stretch
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (channels + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                   (output_time + threadsPerBlock.z - 1) / threadsPerBlock.z);

    time_stretch_kernel_fp16<<<numBlocks, threadsPerBlock>>>(d_temp, d_output, batch_size, channels, input_time, output_time, time_stretch_factor);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * output_time * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
}

} // extern "C"
