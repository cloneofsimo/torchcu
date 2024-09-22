
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

__global__ void interpolate_linear_kernel(const float* input, float* output, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size) {
        float ratio = (float)idx / (output_size - 1);
        int left_idx = (int)(ratio * (input_size - 1));
        int right_idx = min(left_idx + 1, input_size - 1);
        float weight = ratio * (input_size - 1) - left_idx;
        output[idx] = (1.0f - weight) * input[left_idx] + weight * input[right_idx];
    }
}

__global__ void zero_crossing_rate_kernel(const float* input, float* output, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_size - 1) {
        output[0] += abs(input[idx + 1] - input[idx]);
    }
}

extern "C" {

void audio_feature_extraction(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_size = va_arg(args, int);
    int sample_rate = va_arg(args, int);
    float* output_tensor = va_arg(args, float*);

    va_end(args);

    // Define target length for interpolation
    int target_length = 16000;

    // Allocate device memory
    float* d_input, *d_interpolated, *d_zcr;
    cudaMalloc(&d_input, input_tensor_size * sizeof(float));
    cudaMalloc(&d_interpolated, target_length * sizeof(float));
    cudaMalloc(&d_zcr, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_size * sizeof(float), cudaMemcpyHostToDevice);

    // Perform linear interpolation on the device
    dim3 threadsPerBlock(256);
    dim3 numBlocks((target_length + threadsPerBlock.x - 1) / threadsPerBlock.x);
    interpolate_linear_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_interpolated, input_tensor_size, target_length);

    // Calculate ZCR on the device
    zero_crossing_rate_kernel<<<1, 256>>>(d_interpolated, d_zcr, target_length);

    // Copy interpolated data and ZCR to host
    cudaMemcpy(output_tensor, d_interpolated, target_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_tensor + target_length, d_zcr, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_interpolated);
    cudaFree(d_zcr);
}

} // extern "C"
