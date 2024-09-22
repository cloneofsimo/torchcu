
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for flattened einsum summation with fp16
__global__ void einsum_kernel_fp16(const half* input_tensor, const half* weight, half* output,
                                        int batch_size, int input_dim, int output_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < batch_size && j < output_dim) {
        half sum = __int2half_rn(0);
        for (int k = 0; k < input_dim; ++k) {
            sum = __hadd(sum, __hmul(input_tensor[i * input_dim + k], weight[k * output_dim + j]));
        }
        output[i * output_dim + j] = sum;
    }
}

extern "C" {

void flatten_einsum_fp16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    half* output = va_arg(args, half*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1 * input_tensor_dim2;
    int output_dim = weight_dim1;

    // Allocate device memory
    half *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(half));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * sizeof(half));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(half));

    // Copy input data to device (converting to fp16)
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);

    einsum_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, batch_size, input_dim, output_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
