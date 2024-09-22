
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

extern "C" {

__global__ void my_function_kernel(const float* input_tensor, float* output_tensor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Convert to fp16
        __half input_fp16 = __float2half_rn(input_tensor[idx]);

        // Non-zero check
        if (input_fp16 != 0) {
            // Clamp to range
            __half clamped_fp16 = __int2half_rn(__float2int_rn(input_fp16) * (1.0f / 32768.0f) * 32767.0f);
            // Update output
            output_tensor[idx] = __half2float(clamped_fp16);
        }
    }
}

void my_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);

    float* output_tensor = va_arg(args, float*);

    va_end(args);

    int size = input_dim0 * input_dim1;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input_tensor, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    my_function_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, size);

    cudaMemcpy(output_tensor, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

}
