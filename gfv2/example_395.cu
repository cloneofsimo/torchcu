
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

__global__ void bucketize_invert_tanh_kernel(const float* input_tensor, const float* buckets, 
                                            half* output_tensor, int* bucketized_tensor, 
                                            int input_size, int bucket_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_size) {
        int bucket_idx = 0;
        for (int i = 0; i < bucket_size; ++i) {
            if (input_tensor[idx] <= buckets[i]) {
                bucket_idx = i;
                break;
            }
        }
        bucketized_tensor[idx] = bucket_idx;
        
        // Invert and apply tanh
        output_tensor[idx] = __float2half_rn(tanhf(1.0f - (float)bucket_idx));
    }
}

extern "C" {

void my_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_size = va_arg(args, int);

    // Extract buckets tensor
    const float* buckets = va_arg(args, const float*);
    int bucket_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    half* output_tensor = va_arg(args, half*);
    int* bucketized_tensor = va_arg(args, int*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_buckets;
    half *d_output_tensor;
    int *d_bucketized_tensor;
    cudaMalloc(&d_input, input_tensor_size * sizeof(float));
    cudaMalloc(&d_buckets, bucket_size * sizeof(float));
    cudaMalloc(&d_output_tensor, input_tensor_size * sizeof(half));
    cudaMalloc(&d_bucketized_tensor, input_tensor_size * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buckets, buckets, bucket_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256; // Adjust this value based on your GPU
    int numBlocks = (input_tensor_size + threadsPerBlock - 1) / threadsPerBlock;

    bucketize_invert_tanh_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_buckets, d_output_tensor, d_bucketized_tensor, input_tensor_size, bucket_size
    );

    // Copy result back to host
    cudaMemcpy(output_tensor, d_output_tensor, input_tensor_size * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(bucketized_tensor, d_bucketized_tensor, input_tensor_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_buckets);
    cudaFree(d_output_tensor);
    cudaFree(d_bucketized_tensor);
}

}  // extern "C"
