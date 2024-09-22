
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

// CUDA kernel for adaptive log softmax
__global__ void adaptive_log_softmax_kernel(const float* input_tensor, float* output_tensor, 
                                            int batch_size, int input_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * input_size) {
        int batch_index = idx / input_size;
        int feature_index = idx % input_size;

        // Calculate the sum of the exp(input) for each batch
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += expf(input_tensor[batch_index * input_size + i]);
        }

        // Calculate the log softmax value
        output_tensor[idx] = input_tensor[idx] - logf(sum);
    }
}

extern "C" {

void adaptive_log_softmax_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract dim
    int dim = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_size = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size * input_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    adaptive_log_softmax_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, input_size, dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
