
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

#include <cutlass/cutlass.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for PReLU with noise injection
__global__ void prelu_noise_kernel_bf16(const float* input_tensor, const float* weight, float* output,
                                        float noise_scale, int batch_size, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * channels) {
        int batch = idx / channels;
        int channel = idx % channels;

        __nv_bfloat16 input_val = float_to_bfloat16(input_tensor[idx]);
        __nv_bfloat16 weight_val = float_to_bfloat16(weight[channel]);

        __nv_bfloat16 output_val = (input_val > 0.0f) ? input_val : input_val * weight_val;
        output_val += float_to_bfloat16(noise_scale) * __int2bfloat16(curand_uniform());

        output[idx] = __bfloat162float(output_val);
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);

    // Extract noise scale
    float noise_scale = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, batch_size * channels * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * sizeof(float));
    cudaMalloc(&d_output, batch_size * channels * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size * channels + threadsPerBlock.x - 1) / threadsPerBlock.x);

    prelu_noise_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, noise_scale, batch_size, channels
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
