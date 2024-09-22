
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

__global__ void mixup_multinomial_norm_kernel(const float* input_tensor, const float* weights, 
                                                float* output, int batch_size, int feature_dim, int num_samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        float sum = 0.0f;
        for (int j = 0; j < num_samples; ++j) {
            int index = i * num_samples + j;
            int sample_index = static_cast<int>(weights[index]);
            
            // Gather samples from the input tensor
            __nv_bfloat16 sample_bf16 = float_to_bfloat16(input_tensor[sample_index * feature_dim + i]);

            // Calculate the mean
            sum += bfloat16_to_float(sample_bf16);
        }

        // Calculate the Frobenius norm
        sum /= num_samples;
        sum *= sum;
        output[i] = sum;
    }
}

extern "C" {

void mixup_multinomial_norm(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weights tensor
    const float* weights = va_arg(args, const float*);
    int weights_dim0 = va_arg(args, int);
    int weights_dim1 = va_arg(args, int);

    // Extract num_samples
    int num_samples = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int feature_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_weights, *d_output;
    cudaMalloc(&d_input, batch_size * feature_dim * sizeof(float));
    cudaMalloc(&d_weights, batch_size * num_samples * sizeof(float));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, batch_size * num_samples * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    mixup_multinomial_norm_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weights, d_output, batch_size, feature_dim, num_samples
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
}

}  // extern "C"