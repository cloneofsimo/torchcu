
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

#define CHECK(x)                                                                \
  do {                                                                       \
    cudaError_t err = (x);                                                   \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                                    \
    }                                                                       \
  } while (0)

extern "C" {

__global__ void spectral_centroid_kernel(
    const float* input, float* spec_centroid,
    int batch_size, int signal_length, int window_size, int hop_length) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) {
        return;
    }

    int frame_count = (signal_length - window_size) / hop_length + 1;
    float* spec_centroid_ptr = spec_centroid + idx * frame_count;
    const float* input_ptr = input + idx * signal_length;

    for (int frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
        float sum = 0.0f;
        float total_energy = 0.0f;
        for (int bin_idx = 0; bin_idx < window_size / 2 + 1; ++bin_idx) {
            int input_idx = frame_idx * hop_length + bin_idx;
            if (input_idx >= signal_length) {
                break;
            }
            float value = input_ptr[input_idx];
            sum += bin_idx * value * value;
            total_energy += value * value;
        }
        if (total_energy > 0.0f) {
            spec_centroid_ptr[frame_idx] = sum / total_energy;
        } else {
            spec_centroid_ptr[frame_idx] = 0.0f;
        }
    }
}

__global__ void rotary_positional_embedding_kernel(
    float* spec_centroid, int batch_size, int frame_count, float sample_rate) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * frame_count) {
        return;
    }

    int frame_idx = idx % frame_count;
    int batch_idx = idx / frame_count;

    float pos = frame_idx;
    float freq = pos * (2 * M_PI / sample_rate);
    float cos_val = cosf(freq);
    float sin_val = sinf(freq);

    spec_centroid[idx * 2] *= cos_val;
    spec_centroid[idx * 2 + 1] *= sin_val;
}

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int signal_length = va_arg(args, int);

    float sample_rate = static_cast<float>(va_arg(args, double));
    int window_size = va_arg(args, int);
    int hop_length = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = 1; // Assuming single batch for now

    // Allocate device memory
    float *d_input, *d_spec_centroid;
    CHECK(cudaMalloc(&d_input, signal_length * batch_size * sizeof(float)));
    CHECK(cudaMalloc(&d_spec_centroid, 
        (signal_length - window_size) / hop_length + 1 * batch_size * 2 * sizeof(float)));

    // Copy input data to device
    CHECK(cudaMemcpy(d_input, input_tensor, signal_length * batch_size * sizeof(float),
              cudaMemcpyHostToDevice));

    // Calculate spectral centroid
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;
    spectral_centroid_kernel<<<num_blocks, block_size>>>(
        d_input, d_spec_centroid, batch_size, signal_length, window_size, hop_length);
    CHECK(cudaDeviceSynchronize());

    // Apply rotary positional embedding
    rotary_positional_embedding_kernel<<<num_blocks, block_size>>>(
        d_spec_centroid, batch_size, (signal_length - window_size) / hop_length + 1, sample_rate);
    CHECK(cudaDeviceSynchronize());

    // Slice the tensor (done on host for simplicity)
    int frame_count = (signal_length - window_size) / hop_length + 1;
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < frame_count / 2; ++j) {
            output[i * frame_count / 2 + j] = d_spec_centroid[i * frame_count * 2 + j * 2];
            output[i * frame_count / 2 + j + frame_count / 2] = d_spec_centroid[i * frame_count * 2 + j * 2 + 1];
        }
    }

    // Free device memory
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_spec_centroid));
}

} // extern "C"
