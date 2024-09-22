
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/util/tensor_view.h"
#include "cutlass/util/tensor_ref.h"

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for Mel-spectrogram calculation
__global__ void mel_spectrogram_kernel(
    const float* audio_tensor, 
    const float* mel_filter, 
    float* mel_spectrogram, 
    int batch_size, 
    int n_fft, 
    int n_mels, 
    int n_frames) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int frame_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int mel_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && frame_idx < n_frames && mel_idx < n_mels) {
        float sum = 0.0f;
        for (int freq_idx = 0; freq_idx < n_fft // 2 + 1; ++freq_idx) {
            // Note: this is assuming n_fft is even. You'll need to adjust this for odd n_fft
            half audio_val = float_to_half(audio_tensor[batch_idx * n_frames * (n_fft // 2 + 1) + frame_idx * (n_fft // 2 + 1) + freq_idx]);
            half mel_val = float_to_half(mel_filter[mel_idx * (n_fft // 2 + 1) + freq_idx]);

            sum += half_to_float(__hmul(audio_val, audio_val)) * half_to_float(mel_val);
        }
        mel_spectrogram[batch_idx * n_mels * n_frames + mel_idx * n_frames + frame_idx] = log1pf(sum);
    }
}

extern "C" {

void torch_mel_spectrogram_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* audio_tensor = va_arg(args, const float*);
    int audio_tensor_dim0 = va_arg(args, int);
    int audio_tensor_dim1 = va_arg(args, int);

    // Extract mel filter tensor
    const float* mel_filter = va_arg(args, const float*);
    int mel_filter_dim0 = va_arg(args, int);
    int mel_filter_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* mel_spectrogram = va_arg(args, float*);

    va_end(args);

    int batch_size = audio_tensor_dim0;
    int n_frames = audio_tensor_dim1 / (mel_filter_dim1 - 1); 
    int n_fft = mel_filter_dim1 * 2 - 1; // Assuming n_fft is even
    int n_mels = mel_filter_dim0;

    // Allocate device memory
    float *d_audio_tensor, *d_mel_filter, *d_mel_spectrogram;
    cudaMalloc(&d_audio_tensor, batch_size * audio_tensor_dim1 * sizeof(float));
    cudaMalloc(&d_mel_filter, mel_filter_dim0 * mel_filter_dim1 * sizeof(float));
    cudaMalloc(&d_mel_spectrogram, batch_size * n_mels * n_frames * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_audio_tensor, audio_tensor, batch_size * audio_tensor_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mel_filter, mel_filter, mel_filter_dim0 * mel_filter_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((n_mels + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n_frames + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

    mel_spectrogram_kernel<<<numBlocks, threadsPerBlock>>>(
        d_audio_tensor, d_mel_filter, d_mel_spectrogram, batch_size, n_fft, n_mels, n_frames
    );

    // Copy result back to host
    cudaMemcpy(mel_spectrogram, d_mel_spectrogram, batch_size * n_mels * n_frames * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_audio_tensor);
    cudaFree(d_mel_filter);
    cudaFree(d_mel_spectrogram);
}

}  // extern "C"
