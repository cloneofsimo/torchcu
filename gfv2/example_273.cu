
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// Reflect padding (kernel)
__global__ void reflect_pad_kernel(const float* input, float* output, int m, int n, int padding) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        int padded_row = row + padding;
        int padded_col = col + padding;
        
        // Reflect padding logic
        if (padded_row < 0) {
            padded_row = -padded_row - 1;
        } else if (padded_row >= m + 2 * padding) {
            padded_row = 2 * (m + 2 * padding - 1) - padded_row;
        }
        if (padded_col < 0) {
            padded_col = -padded_col - 1;
        } else if (padded_col >= n + 2 * padding) {
            padded_col = 2 * (n + 2 * padding - 1) - padded_col;
        }
        
        output[row * (n + 2 * padding) + col] = input[padded_row * n + padded_col];
    }
}

// Element-wise maximum with weight (kernel)
__global__ void elementwise_max_kernel(const float* input, const float* weights, float* output, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m * n) {
        output[idx] = fmaxf(input[idx], weights[idx]);
    }
}

// Gather operation (kernel)
__global__ void gather_kernel(const float* input, const int* gather_index, float* output, int m, int n, int num_gather) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_gather) {
        int gather_row = gather_index[idx];
        output[idx] = input[gather_row * n + idx]; 
    }
}

// Wasserstein distance calculation (kernel)
__global__ void wasserstein_dist_kernel(const float* gathered, const float* input, float* output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += fabsf(gathered[row * n + i] - input[col * n + i]);
        }
        output[row * n + col] = sum;
    }
}

extern "C" {

void custom_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);
    const float* weights = va_arg(args, const float*);
    int weights_dim0 = va_arg(args, int);
    int weights_dim1 = va_arg(args, int);
    const int padding = va_arg(args, int);
    const int* gather_index = va_arg(args, const int*);
    int gather_index_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_weights, *d_padded, *d_max_tensor, *d_gathered, *d_output;
    cudaMalloc(&d_input, input_dim0 * input_dim1 * sizeof(float));
    cudaMalloc(&d_weights, weights_dim0 * weights_dim1 * sizeof(float));
    cudaMalloc(&d_padded, input_dim0 * (input_dim1 + 2 * padding) * sizeof(float));
    cudaMalloc(&d_max_tensor, input_dim0 * (input_dim1 + 2 * padding) * sizeof(float));
    cudaMalloc(&d_gathered, gather_index_dim0 * input_dim1 * sizeof(float));
    cudaMalloc(&d_output, input_dim0 * input_dim1 * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, input_dim0 * input_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, weights_dim0 * weights_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gather_index, gather_index, gather_index_dim0 * sizeof(int), cudaMemcpyHostToDevice);

    // Reflect Padding
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim1 + 2 * padding + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (input_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);
    reflect_pad_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_padded, input_dim0, input_dim1, padding);

    // Element-wise maximum
    numBlocks = ((input_dim0 * (input_dim1 + 2 * padding)) + threadsPerBlock.x - 1) / threadsPerBlock.x;
    elementwise_max_kernel<<<numBlocks, threadsPerBlock>>>(d_padded, d_weights, d_max_tensor, input_dim0, input_dim1 + 2 * padding);

    // Gather
    numBlocks = (gather_index_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    gather_kernel<<<numBlocks, threadsPerBlock>>>(d_max_tensor, d_gather_index, d_gathered, input_dim0, input_dim1 + 2 * padding, gather_index_dim0);

    // Wasserstein Distance
    numBlocks = (input_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    wasserstein_dist_kernel<<<numBlocks, threadsPerBlock>>>(d_gathered, d_input, d_output, input_dim0, input_dim1);

    // Copy result back to host
    cudaMemcpy(output, d_output, input_dim0 * input_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_padded);
    cudaFree(d_max_tensor);
    cudaFree(d_gathered);
    cudaFree(d_output);
}

}  // extern "C"
