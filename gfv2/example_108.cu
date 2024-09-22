
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper function for efficient pairwise distance calculation
__global__ void pairwise_distance_kernel(const float* input, float* output, int batch_size, int time_steps, int features) {
    int b = blockIdx.x;
    int i = threadIdx.x;
    int j = threadIdx.y;

    if (b < batch_size && i < time_steps && j < time_steps) {
        float sum = 0.0f;
        for (int k = 0; k < features; k++) {
            float diff = input[b * time_steps * features + i * features + k] - input[b * time_steps * features + j * features + k];
            sum += diff * diff;
        }
        output[b * time_steps * time_steps + i * time_steps + j] = sum;
    }
}

// Helper function for efficient time stretching with constant padding
__global__ void time_stretch_constant_pad_kernel(const float* input, float* output, int batch_size, int in_time_steps, 
                                               int out_time_steps, int features, float stretch_factor, float pad_value) {
    int b = blockIdx.x;
    int t = threadIdx.x;
    int f = threadIdx.y;

    if (b < batch_size && t < out_time_steps && f < features) {
        // Calculate the corresponding index in the input tensor
        int input_t = min(int(t / stretch_factor), in_time_steps - 1);

        // Copy the value if it's within the input bounds, otherwise use the padding value
        if (input_t >= 0) {
            output[b * out_time_steps * features + t * features + f] = input[b * in_time_steps * features + input_t * features + f];
        } else {
            output[b * out_time_steps * features + t * features + f] = pad_value;
        }
    }
}

extern "C" {

void time_stretch_pairwise_distance(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int in_time_steps = va_arg(args, int);
    int features = va_arg(args, int);

    // Extract time stretch factor
    float stretch_factor = va_arg(args, double);

    // Extract padding mode
    char* pad_mode = va_arg(args, char*);

    // Extract padding value
    float pad_value = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Calculate the output time steps based on the stretch factor
    int out_time_steps = int(in_time_steps * stretch_factor);

    // Allocate device memory for stretched input tensor
    float* d_stretched_input;
    cudaMalloc(&d_stretched_input, batch_size * out_time_steps * features * sizeof(float));

    // Launch kernel for time stretching with constant padding
    dim3 threadsPerBlock(32, 8);
    dim3 numBlocks(batch_size, 1);
    time_stretch_constant_pad_kernel<<<numBlocks, threadsPerBlock>>>(input, d_stretched_input, batch_size, in_time_steps, out_time_steps, features, stretch_factor, pad_value);

    // Launch kernel for pairwise distance calculation
    dim3 threadsPerBlock2(16, 16);
    dim3 numBlocks2(batch_size, 1);
    pairwise_distance_kernel<<<numBlocks2, threadsPerBlock2>>>(d_stretched_input, output, batch_size, out_time_steps, features);

    // Free device memory
    cudaFree(d_stretched_input);
}

} // extern "C"
