
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __half
__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert __half to float
__device__ __forceinline__ float half_to_float(__half h) {
    return __half2float(h);
}

__global__ void hadamard_product_repeat_kthvalue_kernel(const float* input_tensor, const float* weight, 
                                                         __half* output, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < 2*m && col < n) {
        // Hadamard product
        __half product = float_to_half(input_tensor[row % m * n + col]) * float_to_half(weight[row % m * n + col]);

        // Repeat along the first dimension
        output[row * n + col] = product; 
    }
}

__global__ void kthvalue_kernel(const __half* input, __half* output, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        // Find the kth value using a simple selection algorithm (can be optimized)
        __half kth_value = input[row * n];
        for (int i = 1; i < n; i++) {
            if (input[row * n + i] < kth_value) {
                kth_value = input[row * n + i];
            }
        }
        output[row * n + col] = kth_value;
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
    int weight_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    __half* output = va_arg(args, __half*);

    va_end(args);

    int m = input_tensor_dim0;
    int n = input_tensor_dim1;
    int k = 3; 

    // Allocate device memory
    float *d_input, *d_weight;
    __half *d_repeated_product, *d_kth_values;
    cudaMalloc(&d_input, m * n * sizeof(float));
    cudaMalloc(&d_weight, m * n * sizeof(float));
    cudaMalloc(&d_repeated_product, 2 * m * n * sizeof(__half));
    cudaMalloc(&d_kth_values, m * n * sizeof(__half));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch hadamard product and repeat kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (2 * m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    hadamard_product_repeat_kthvalue_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_repeated_product, m, n, k);

    // Launch kth value kernel
    numBlocks = ((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    kthvalue_kernel<<<numBlocks, threadsPerBlock>>>(d_repeated_product, d_kth_values, m, n, k);

    // Copy result back to host
    cudaMemcpy(output, d_kth_values, m * n * sizeof(__half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_repeated_product);
    cudaFree(d_kth_values);
}
}
