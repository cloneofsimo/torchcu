
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdarg.h>

#define CHECK(x)                                                                       \
    {                                                                               \
        cudaError_t err = (x);                                                      \
        if (err != cudaSuccess) {                                                   \
            fprintf(stderr, "Error %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                               \
    }

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

__global__ void spectrogram_bandwidth_kernel(
    const float* audio,
    const int sample_rate,
    const int window_size,
    const int hop_size,
    float* bandwidth,
    const int batch_size,
    const int audio_length) {
    const int time_steps = (audio_length - window_size) / hop_size + 1;
    const int num_freq_bins = window_size / 2 + 1;

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int time_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch_size && time_idx < time_steps) {
        int start_idx = time_idx * hop_size;
        int end_idx = start_idx + window_size;
        int offset = batch_idx * audio_length;

        // Calculate the spectrogram
        float window_sum = 0.0f;
        half complex_sum[num_freq_bins];
        for (int i = 0; i < num_freq_bins; i++) {
            complex_sum[i] = make_half2(0.0f, 0.0f);
        }

        for (int t = start_idx; t < end_idx; t++) {
            float window_value = 0.5f * (1.0f - cosf(2.0f * M_PI * t / (window_size - 1))); // Hann window
            window_sum += window_value;
            for (int i = 0; i < num_freq_bins; i++) {
                float real = audio[offset + t] * cosf(2.0f * M_PI * i * t / window_size) * window_value;
                float imag = audio[offset + t] * sinf(2.0f * M_PI * i * t / window_size) * window_value;
                complex_sum[i].x += float_to_half(real);
                complex_sum[i].y += float_to_half(imag);
            }
        }
        for (int i = 0; i < num_freq_bins; i++) {
            complex_sum[i].x /= float_to_half(window_sum);
            complex_sum[i].y /= float_to_half(window_sum);
        }

        // Calculate squared magnitude spectrogram
        float spectrogram[num_freq_bins];
        for (int i = 0; i < num_freq_bins; i++) {
            spectrogram[i] = half_to_float(complex_sum[i].x * complex_sum[i].x +
                                         complex_sum[i].y * complex_sum[i].y);
        }

        // Calculate spectral bandwidth
        float spectrogram_sum = 0.0f;
        float mean_frequency = 0.0f;
        float squared_deviation_sum = 0.0f;
        for (int i = 0; i < num_freq_bins; i++) {
            float frequency = i * (float)sample_rate / window_size;
            spectrogram_sum += spectrogram[i];
            mean_frequency += spectrogram[i] * frequency;
            squared_deviation_sum += spectrogram[i] * (frequency - mean_frequency) * (frequency - mean_frequency);
        }
        mean_frequency /= spectrogram_sum;
        float bandwidth_value = sqrtf(squared_deviation_sum / spectrogram_sum);
        bandwidth[batch_idx * time_steps + time_idx] = bandwidth_value;
    }
}

extern "C" {
void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* audio = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int audio_length = va_arg(args, int);

    int sample_rate = va_arg(args, int);
    int window_size = va_arg(args, int);
    int hop_size = va_arg(args, int);

    float* bandwidth = va_arg(args, float*);

    va_end(args);

    const int time_steps = (audio_length - window_size) / hop_size + 1;
    const int num_freq_bins = window_size / 2 + 1;

    // Allocate device memory
    float* d_audio;
    float* d_bandwidth;

    CHECK(cudaMalloc(&d_audio, batch_size * audio_length * sizeof(float)));
    CHECK(cudaMalloc(&d_bandwidth, batch_size * time_steps * sizeof(float)));

    // Copy data to device
    CHECK(cudaMemcpy(d_audio, audio, batch_size * audio_length * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel
    dim3 threadsPerBlock(32, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (time_steps + threadsPerBlock.y - 1) / threadsPerBlock.y);
    spectrogram_bandwidth_kernel<<<numBlocks, threadsPerBlock>>>(
        d_audio, sample_rate, window_size, hop_size, d_bandwidth, batch_size, audio_length);

    // Copy result back to host
    CHECK(cudaMemcpy(bandwidth, d_bandwidth, batch_size * time_steps * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(d_audio));
    CHECK(cudaFree(d_bandwidth));
}
}
