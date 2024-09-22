
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h> 

// Helper function for int8 hardtanh
__device__ __forceinline__ int8_t int8_hardtanh(int8_t x, int8_t min_val, int8_t max_val) {
    return (x < min_val) ? min_val : ((x > max_val) ? max_val : x);
}

__global__ void quantized_bucketing_with_gradient_clipping_kernel(const float* input, const int* buckets, int* output, 
                                                                 int size, int num_buckets, float gradient_clip_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int8_t input_int8 = __int_as_int8(input[idx]); 
        int bucket_idx = 0;
        
        // Find the correct bucket
        for (int i = 0; i < num_buckets; i++) {
            if (input_int8 <= buckets[i]) {
                bucket_idx = i;
                break;
            }
        }

        // Hardtanh with gradient clipping
        int8_t clipped_value = int8_hardtanh(bucket_idx, __int_as_int8(-gradient_clip_value), __int_as_int8(gradient_clip_value));

        output[idx] = clipped_value;
    }
}

extern "C" {

void quantized_bucketing_with_gradient_clipping(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int input_size = va_arg(args, int);

    // Extract buckets list
    const int* buckets = va_arg(args, const int*);
    int num_buckets = va_arg(args, int);

    // Extract gradient clip value
    float gradient_clip_value = va_arg(args, double);

    // Extract output tensor
    int* output = va_arg(args, int*);

    va_end(args);

    // Allocate device memory
    int *d_buckets, *d_output;
    cudaMalloc(&d_buckets, num_buckets * sizeof(int));
    cudaMalloc(&d_output, input_size * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_buckets, buckets, num_buckets * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((input_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    quantized_bucketing_with_gradient_clipping_kernel<<<numBlocks, threadsPerBlock>>>(
        input, d_buckets, d_output, input_size, num_buckets, gradient_clip_value
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, input_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_buckets);
    cudaFree(d_output);
}

} // extern "C"
