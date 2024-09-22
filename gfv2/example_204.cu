
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>

// Helper function to convert float to __half
__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert __half to float
__device__ __forceinline__ float half_to_float(__half h) {
    return __half2float(h);
}

// CUDA kernel for exp, cumsum, and matrix_exp
__global__ void exp_cumsum_matrix_exp_kernel(const float* input_tensor, float* output, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        // Exp
        __half exp_val = expf(float_to_half(input_tensor[row * n * k + col]));
        output[row * n * k + col] = half_to_float(exp_val);

        // Cumsum
        for (int i = row + 1; i < m; ++i) {
            output[i * n * k + col] += output[(i - 1) * n * k + col];
        }

        // Matrix Exp (Naive implementation - can be optimized)
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += output[row * n * k + col + i * n];
        }
        output[row * n * k + col] = expf(sum);
    }
}

extern "C" {

void exp_cumsum_matrix_exp_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int rows = input_tensor_dim1;
    int cols = input_tensor_dim2;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * rows * cols * sizeof(float));
    cudaMalloc(&d_output, batch_size * rows * cols * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    exp_cumsum_matrix_exp_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, rows, cols
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

} // extern "C"
