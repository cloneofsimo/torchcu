
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for audio resynthesis
__global__ void audio_resynthesis_kernel_bf16(const float* spectrogram, const float* phase, 
                                              float* output, int batch_size, int time_steps, 
                                              int freq_bins, int hop_length) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int time_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch_size && time_idx < time_steps) {
        float sum = 0.0f;

        for (int freq_idx = 0; freq_idx < freq_bins; ++freq_idx) {
            int spectrogram_idx = batch_idx * time_steps * freq_bins + time_idx * freq_bins + freq_idx;
            int phase_idx = spectrogram_idx;

            // Load spectrogram and phase values in bfloat16
            __nv_bfloat16 mag = float_to_bfloat16(spectrogram[spectrogram_idx]);
            __nv_bfloat16 angle = float_to_bfloat16(phase[phase_idx]);

            // Calculate complex exponential in bfloat16
            __nv_bfloat16 complex_value = __hmul(mag, __expf(angle * 1j));
            
            // Convert back to float and add to sum (this is the iSTFT operation)
            sum += bfloat16_to_float(complex_value); 
        }

        // Output the final resynthesized audio value
        output[batch_idx * time_steps + time_idx] = sum;
    }
}

extern "C" {

void audio_resynthesis_bf16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* spectrogram = va_arg(args, const float*);
    int spectrogram_dim0 = va_arg(args, int);
    int spectrogram_dim1 = va_arg(args, int);
    int spectrogram_dim2 = va_arg(args, int);

    const float* phase = va_arg(args, const float*);
    int phase_dim0 = va_arg(args, int);
    int phase_dim1 = va_arg(args, int);
    int phase_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = spectrogram_dim0;
    int time_steps = spectrogram_dim1;
    int freq_bins = spectrogram_dim2;
    int hop_length = 512;  // Assuming hop length is 512

    // Allocate device memory
    float *d_spectrogram, *d_phase, *d_output;
    cudaMalloc(&d_spectrogram, batch_size * time_steps * freq_bins * sizeof(float));
    cudaMalloc(&d_phase, batch_size * time_steps * freq_bins * sizeof(float));
    cudaMalloc(&d_output, batch_size * time_steps * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_spectrogram, spectrogram, batch_size * time_steps * freq_bins * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phase, phase, batch_size * time_steps * freq_bins * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (time_steps + threadsPerBlock.y - 1) / threadsPerBlock.y);

    audio_resynthesis_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_spectrogram, d_phase, d_output, batch_size, time_steps, freq_bins, hop_length
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * time_steps * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_spectrogram);
    cudaFree(d_phase);
    cudaFree(d_output);
}

}  // extern "C"
