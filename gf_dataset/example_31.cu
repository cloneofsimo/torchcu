
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

__global__ void scatter_add_fp16_kernel(const float* input_tensor, const int* indices, float* output_tensor, 
                                         int batch_size, int input_dim, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        int index = indices[idx];
        int offset = index * input_dim + dim; 

        if (offset >= 0 && offset < batch_size * input_dim) {
            output_tensor[offset] = half_to_float(
                __hadd(float_to_half(output_tensor[offset]), float_to_half(input_tensor[idx * input_dim + dim]))
            );
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

    // Extract indices tensor
    const int* indices = va_arg(args, const int*);

    // Extract dim
    int dim = va_arg(args, int);

    // Extract output tensor
    float* output_tensor = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    int *d_indices;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_indices, batch_size * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    scatter_add_fp16_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_indices, d_output, batch_size, input_dim, dim);

    // Copy result back to host
    cudaMemcpy(output_tensor, d_output, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);
}
}
