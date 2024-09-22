
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void zero_crossing_rate_kernel(const int8_t* input_tensor, float* output,
                                        int batch_size, int channels, int length) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && c < channels) {
        float zcr = 0.0f;
        for (int i = 1; i < length; ++i) {
            if ((input_tensor[b * channels * length + c * length + i] * 
                 input_tensor[b * channels * length + c * length + i - 1]) < 0) {
                zcr += 1.0f;
            }
        }
        output[b * channels + c] = zcr / (length - 1);
    }
}

extern "C" {

void zero_crossing_rate_int8(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor_float = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int channels = va_arg(args, int);
    int length = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory for input and output
    int8_t* d_input;
    cudaMalloc(&d_input, batch_size * channels * length * sizeof(int8_t));

    // Copy input data to device as int8
    cudaMemcpy(d_input, input_tensor_float, batch_size * channels * length * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (channels + threadsPerBlock.y - 1) / threadsPerBlock.y);

    zero_crossing_rate_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, output, batch_size, channels, length
    );

    // Free device memory
    cudaFree(d_input);
}

}  // extern "C"
