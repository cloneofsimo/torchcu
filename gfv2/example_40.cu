
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end
#include <curand_kernel.h> // for curand
#include <cutlass/cutlass.h> // for cutlass

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for uniform distribution and backward pass using bfloat16
__global__ void uniform_backward_kernel_bf16(const float* input_tensor, float* output_grad,
                                        int m, int n, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        // Generate random number in bfloat16
        curandState_t state;
        curand_init(row * n + col, 0, 0, &state);
        __nv_bfloat16 random_value = float_to_bfloat16(curand_uniform(&state));
        __nv_bfloat16 scaled_value = __hmul(random_value, float_to_bfloat16(scale));

        // Apply backward pass
        output_grad[row * n + col] = 1.0f; // Assuming output.backward(torch.ones_like(...))
    }
}

extern "C" {

void bfloat16_uniform_backward(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract scale
    float scale = va_arg(args, float);

    // Extract output gradient tensor (assuming it's preallocated)
    float* output_grad = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output_grad;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output_grad, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    uniform_backward_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output_grad, batch_size, input_dim, scale
    );

    // Copy result back to host
    cudaMemcpy(output_grad, d_output_grad, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output_grad);
}

}  // extern "C"
