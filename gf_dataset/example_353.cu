
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

__global__ void gradient_magnitude_bf16_kernel(const float* input_tensor, float* output, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        __nv_bfloat16 left = float_to_bfloat16(input_tensor[(row * width) + (col - 1)]);
        __nv_bfloat16 right = float_to_bfloat16(input_tensor[(row * width) + (col + 1)]);
        __nv_bfloat16 up = float_to_bfloat16(input_tensor[((row - 1) * width) + col]);
        __nv_bfloat16 down = float_to_bfloat16(input_tensor[((row + 1) * width) + col]);

        __nv_bfloat16 grad_x = float_to_bfloat16(abs(bfloat16_to_float(right) - bfloat16_to_float(left)));
        __nv_bfloat16 grad_y = float_to_bfloat16(abs(bfloat16_to_float(down) - bfloat16_to_float(up)));

        __nv_bfloat16 magnitude = float_to_bfloat16(sqrt(bfloat16_to_float(grad_x) * bfloat16_to_float(grad_x) +
                                                         bfloat16_to_float(grad_y) * bfloat16_to_float(grad_y)));

        output[(row * width) + col] = bfloat16_to_float(magnitude);
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int height = va_arg(args, int);
    int width = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, height * width * sizeof(float));
    cudaMalloc(&d_output, height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gradient_magnitude_bf16_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, height, width
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
