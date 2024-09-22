
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

// CUDA kernel for mean calculation (using shared memory for better coalesced access)
__global__ void mean_kernel(const float* input, float* mean, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < d) {
        float sum = 0.0f;
        int k = j * n + i;
        sum += input[k];
        __shared__ float smem[256];
        smem[threadIdx.x + threadIdx.y * blockDim.x] = sum;
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            sum = 0.0f;
            for (int k = 0; k < blockDim.x * blockDim.y; k++) {
                sum += smem[k];
            }
            mean[j] = sum / (n * d);
        }
    }
}

// CUDA kernel for standard deviation calculation (using shared memory)
__global__ void stddev_kernel(const float* input, const float* mean, float* stddev, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < d) {
        float sum_sq = 0.0f;
        int k = j * n + i;
        sum_sq += (input[k] - mean[j]) * (input[k] - mean[j]);
        __shared__ float smem[256];
        smem[threadIdx.x + threadIdx.y * blockDim.x] = sum_sq;
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            sum_sq = 0.0f;
            for (int k = 0; k < blockDim.x * blockDim.y; k++) {
                sum_sq += smem[k];
            }
            stddev[j] = sqrtf(sum_sq / (n * d));
        }
    }
}

// CUDA kernel for spectral contrast normalization (using shared memory for broadcast)
__global__ void spec_contrast_kernel(float* input, const float* mean, const float* stddev, float scale, float bias, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < d) {
        int k = j * n + i;
        __shared__ float s_mean[256], s_stddev[256];
        s_mean[threadIdx.x + threadIdx.y * blockDim.x] = mean[j];
        s_stddev[threadIdx.x + threadIdx.y * blockDim.x] = stddev[j];
        __syncthreads();
        input[k] = __float2bfloat16(fabsf(__bfloat162float(scale * ((__float2bfloat16(input[k]) - s_mean[threadIdx.x + threadIdx.y * blockDim.x]) / s_stddev[threadIdx.x + threadIdx.y * blockDim.x])) + bias)); 
    }
}

extern "C" {

void spectral_contrast_inplace_bf16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    float* input_tensor = va_arg(args, float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract scale
    float scale = va_arg(args, float);

    // Extract bias
    float bias = va_arg(args, float);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float* d_input;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate mean
    float* d_mean = new float[input_dim];
    float* h_mean = new float[input_dim];
    cudaMalloc(&d_mean, input_dim * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, (input_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);
    mean_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_mean, batch_size, input_dim);
    cudaMemcpy(h_mean, d_mean, input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate standard deviation
    float* d_stddev = new float[input_dim];
    float* h_stddev = new float[input_dim];
    cudaMalloc(&d_stddev, input_dim * sizeof(float));
    stddev_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_mean, d_stddev, batch_size, input_dim);
    cudaMemcpy(h_stddev, d_stddev, input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Apply spectral contrast normalization
    spec_contrast_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_mean, d_stddev, scale, bias, batch_size, input_dim);

    // Copy result back to host
    cudaMemcpy(input_tensor, d_input, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_mean);
    cudaFree(d_stddev);
    delete[] h_mean;
    delete[] h_stddev;
}

}  // extern "C"
