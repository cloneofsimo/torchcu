
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

__global__ void scatter_kernel_bf16(const float* input_tensor, const int* index, float* output, 
                                        int size, int dim, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        int idx = index[i];
        if (idx >= 0 && idx < size) {
            int offset = idx;
            if (dim != 0) {
                offset += i / size * size;
            }
            output[offset] = input_tensor[i];
        }
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

    // Extract index tensor
    const int* index = va_arg(args, const int*);
    int index_dim0 = va_arg(args, int);

    // Extract dim
    int dim = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int num_elements = input_tensor_dim0 * input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    int *d_index;
    cudaMalloc(&d_input, num_elements * sizeof(float));
    cudaMalloc(&d_index, index_dim0 * sizeof(int));
    cudaMalloc(&d_output, num_elements * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, index, index_dim0 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((num_elements + threadsPerBlock.x - 1) / threadsPerBlock.x);
    
    scatter_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_index, d_output, input_tensor_dim0, dim, num_elements
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_index);
    cudaFree(d_output);
}

}  // extern "C"
