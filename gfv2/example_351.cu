
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

extern "C" {

__global__ void linspace_floor_fp16_kernel(const float* input_tensor, float start, float end, int steps, half* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        float value = start + (end - start) * idx / (steps - 1);
        output[idx] = __float2half_rn(floorf(value));
    }
}

void linspace_floor_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    float start = va_arg(args, float);
    float end = va_arg(args, float);
    int steps = va_arg(args, int);

    half* output;
    cudaMalloc(&output, steps * sizeof(half));

    linspace_floor_fp16_kernel<<<(steps + 255) / 256, 256>>>(input_tensor, start, end, steps, output);

    cudaMemcpy(input_tensor, output, steps * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(output);
}
}
