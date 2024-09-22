
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

// CUDA kernel for bmm
__global__ void bmm_kernel_bf16(const float* input1, const float* input2, float* output1,
                                 int N, int C, int H, int W, int M, int K) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < N && row < M && col < W) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input1[batch_idx * C * H * W + i * H * W + row * W + col]);
            __nv_bfloat16 b = float_to_bfloat16(input2[col * K + i]);
            sum += bfloat16_to_float(__hmul(a, b));
        }
        output1[batch_idx * M * H * W + row * W + col] = sum;
    }
}

// CUDA kernel for baddbmm
__global__ void baddbmm_kernel_bf16(const float* input3, const float* input1, const float* input2, float* output2,
                                  int N, int C, int H, int W, int M, int K) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < N && row < C && col < W) {
        float sum = input3[batch_idx * C * H * W + row * W + col];
        for (int i = 0; i < K; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input1[batch_idx * C * H * W + i * H * W + row * W + col]);
            __nv_bfloat16 b = float_to_bfloat16(input2[col * K + i]);
            sum += bfloat16_to_float(__hmul(a, b));
        }
        output2[batch_idx * C * H * W + row * W + col] = sum;
    }
}

extern "C" {

void complex_bfloat16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input1 = va_arg(args, const float*);
    int input1_dim0 = va_arg(args, int);
    int input1_dim1 = va_arg(args, int);
    int input1_dim2 = va_arg(args, int);
    int input1_dim3 = va_arg(args, int);

    const float* input2 = va_arg(args, const float*);
    int input2_dim0 = va_arg(args, int);
    int input2_dim1 = va_arg(args, int);

    const float* input3 = va_arg(args, const float*);
    int input3_dim0 = va_arg(args, int);
    int input3_dim1 = va_arg(args, int);
    int input3_dim2 = va_arg(args, int);
    int input3_dim3 = va_arg(args, int);

    // Extract output tensors (assuming they're preallocated)
    float* output1 = va_arg(args, float*);
    float* output2 = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input1, *d_input2, *d_input3, *d_output1, *d_output2;
    cudaMalloc(&d_input1, input1_dim0 * input1_dim1 * input1_dim2 * input1_dim3 * sizeof(float));
    cudaMalloc(&d_input2, input2_dim0 * input2_dim1 * sizeof(float));
    cudaMalloc(&d_input3, input3_dim0 * input3_dim1 * input3_dim2 * input3_dim3 * sizeof(float));
    cudaMalloc(&d_output1, input1_dim0 * input2_dim0 * input1_dim2 * input1_dim3 * sizeof(float));
    cudaMalloc(&d_output2, input3_dim0 * input3_dim1 * input3_dim2 * input3_dim3 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input1, input1, input1_dim0 * input1_dim1 * input1_dim2 * input1_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2, input2_dim0 * input2_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input3, input3, input3_dim0 * input3_dim1 * input3_dim2 * input3_dim3 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch bmm kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((input1_dim3 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input2_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (input1_dim0 + threadsPerBlock.z - 1) / threadsPerBlock.z);

    bmm_kernel_bf16<<<numBlocks, threadsPerBlock>>>(d_input1, d_input2, d_output1,
                                                  input1_dim0, input1_dim1, input1_dim2, input1_dim3,
                                                  input2_dim0, input2_dim1);

    // Launch baddbmm kernel
    numBlocks = ((input3_dim3 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (input3_dim1 + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (input3_dim0 + threadsPerBlock.z - 1) / threadsPerBlock.z);

    baddbmm_kernel_bf16<<<numBlocks, threadsPerBlock>>>(d_input3, d_input1, d_input2, d_output2,
                                                    input3_dim0, input3_dim1, input3_dim2, input3_dim3,
                                                    input2_dim0, input2_dim1);

    // Copy results back to host
    cudaMemcpy(output1, d_output1, input1_dim0 * input2_dim0 * input1_dim2 * input1_dim3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output2, d_output2, input3_dim0 * input3_dim1 * input3_dim2 * input3_dim3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_input3);
    cudaFree(d_output1);
    cudaFree(d_output2);
}

}  // extern "C"
