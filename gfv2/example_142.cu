
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

// CUDA kernel for block diagonal matrix creation
__global__ void create_block_diag_kernel(const float* input, float* output, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        if (row == col) {
            output[row * n + col] = input[row * k];
        } else {
            output[row * n + col] = 0.0f;
        }
    }
}

// CUDA kernel for exponential and grid multiplication
__global__ void exp_grid_mul_kernel(const float* input1, const float* input2, const float* grid, float* output,
                                    int n, int k, int m, int l) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < m) {
        __nv_bfloat16 sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input1[row * k + i]);
            __nv_bfloat16 b = float_to_bfloat16(input2[row * k + i]);
            __nv_bfloat16 c = float_to_bfloat16(grid[col * l + i]);
            sum = __hmul(a, b, c);
        }
        output[row * m + col] = bfloat16_to_float(sum);
    }
}

// CUDA kernel for ReLU activation
__global__ void relu_kernel(float* output, int n, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < m) {
        output[row * m + col] = fmaxf(output[row * m + col], 0.0f);
    }
}

extern "C" {

void my_tensor_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor1 = va_arg(args, const float*);
    int input_tensor1_dim0 = va_arg(args, int);
    int input_tensor1_dim1 = va_arg(args, int);
    int input_tensor1_dim2 = va_arg(args, int);
    int input_tensor1_dim3 = va_arg(args, int);

    const float* input_tensor2 = va_arg(args, const float*);
    int input_tensor2_dim0 = va_arg(args, int);
    int input_tensor2_dim1 = va_arg(args, int);
    int input_tensor2_dim2 = va_arg(args, int);
    int input_tensor2_dim3 = va_arg(args, int);

    const float* input_tensor3 = va_arg(args, const float*);
    int input_tensor3_dim0 = va_arg(args, int);
    int input_tensor3_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output_tensor = va_arg(args, float*);

    va_end(args);

    int n = input_tensor1_dim0;
    int k = input_tensor1_dim1 * input_tensor1_dim2 * input_tensor1_dim3;
    int m = input_tensor3_dim0;
    int l = input_tensor3_dim1;

    // Allocate device memory
    float *d_input1, *d_input2, *d_input3, *d_block_diag, *d_output;
    cudaMalloc(&d_input1, n * k * sizeof(float));
    cudaMalloc(&d_input2, n * k * sizeof(float));
    cudaMalloc(&d_input3, m * l * sizeof(float));
    cudaMalloc(&d_block_diag, n * n * sizeof(float));
    cudaMalloc(&d_output, n * m * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input1, input_tensor1, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input_tensor2, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input3, input_tensor3, m * l * sizeof(float), cudaMemcpyHostToDevice);

    // 1. Block Diagonal
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    create_block_diag_kernel<<<numBlocks, threadsPerBlock>>>(d_input1, d_block_diag, n, k);

    // 2. Exponential and Grid Multiplication (using bfloat16)
    dim3 threadsPerBlock2(16, 16);
    dim3 numBlocks2((m + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
                    (n + threadsPerBlock2.y - 1) / threadsPerBlock2.y);
    exp_grid_mul_kernel<<<numBlocks2, threadsPerBlock2>>>(
        d_block_diag, d_input2, d_input3, d_output, n, k, m, l
    );

    // 3. ReLU activation
    relu_kernel<<<numBlocks2, threadsPerBlock2>>>(d_output, n, m);

    // Copy result back to host
    cudaMemcpy(output_tensor, d_output, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_input3);
    cudaFree(d_block_diag);
    cudaFree(d_output);
}

}  // extern "C"
