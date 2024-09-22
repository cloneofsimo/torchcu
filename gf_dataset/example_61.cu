
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for scatter add, softsign, and layer norm
__global__ void scatter_add_softsign_layer_norm_kernel(const float* input_tensor, const int* indices, const float* weight, float* output,
                                                        int batch_size, int feature_dim, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        int i = indices[idx];
        if (i >= 0 && i < batch_size) {
            float sum = 0.0f;
            for (int j = 0; j < feature_dim; j++) {
                sum += input_tensor[i * feature_dim + j];
            }
            for (int j = 0; j < feature_dim; j++) {
                output[idx * feature_dim + j] = sum * weight[j];
            }
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
    int indices_dim0 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int feature_dim = input_tensor_dim1;
    int num_elements = indices_dim0;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    int *d_indices;
    cudaMalloc(&d_input, batch_size * feature_dim * sizeof(float));
    cudaMalloc(&d_indices, num_elements * sizeof(int));
    cudaMalloc(&d_weight, feature_dim * sizeof(float));
    cudaMalloc(&d_output, num_elements * feature_dim * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, feature_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((num_elements + threadsPerBlock.x - 1) / threadsPerBlock.x);

    scatter_add_softsign_layer_norm_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_indices, d_weight, d_output, batch_size, feature_dim, num_elements
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, num_elements * feature_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_indices);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
