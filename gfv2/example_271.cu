
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

// This version uses shared memory for better performance.

__global__ void low_rank_approx_pow_kernel(const float* input_tensor, float* output_tensor, 
                                             int m, int n, int rank, float exponent) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        __shared__ float U_shared[128]; 
        __shared__ float V_shared[128]; 

        // Load U and V sub-blocks into shared memory
        if (threadIdx.x < rank && threadIdx.y < rank) {
            U_shared[threadIdx.x * blockDim.y + threadIdx.y] = input_tensor[row * n + threadIdx.x];
            V_shared[threadIdx.y * blockDim.x + threadIdx.x] = input_tensor[threadIdx.y * n + col]; 
        }

        __syncthreads();

        // Calculate the sum for the output element
        for (int k = 0; k < rank; k++) {
            sum += U_shared[threadIdx.y * blockDim.y + k] * V_shared[k * blockDim.x + threadIdx.x];
        }

        output_tensor[row * n + col] = powf(sum, exponent); 
    }
}

extern "C" {

void low_rank_approx_pow(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int m = va_arg(args, int);
    int n = va_arg(args, int);

    int rank = va_arg(args, int);
    float exponent = va_arg(args, float);

    float* output_tensor = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float* d_input, *d_output;
    cudaMalloc(&d_input, m * n * sizeof(float));
    cudaMalloc(&d_output, m * n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    low_rank_approx_pow_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, m, n, rank, exponent
    );

    // Copy result back to host
    cudaMemcpy(output_tensor, d_output, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
