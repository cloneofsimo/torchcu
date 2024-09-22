
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

// Helper function to convert float to __half
__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert __half to float
__device__ __forceinline__ float half_to_float(__half hf) {
    return __half2float(hf);
}

// CUDA kernel for the complex tensor operation
__global__ void complex_tensor_operation_kernel(const float* input_tensor, const float* target_values, float* output, 
                                                int input_tensor_size, int target_values_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < input_tensor_size) {
        __nv_bfloat16 input_value = float_to_bfloat16(input_tensor[i]);
        __half mask_value = 0.0h;

        // Check if input_value is present in target_values
        for (int j = 0; j < target_values_size; ++j) {
            if (input_value == float_to_bfloat16(target_values[j])) {
                mask_value = 1.0h;
                break;
            }
        }

        // Perform matrix multiplication with mask
        __half sum = 0.0h;
        sum += float_to_half(input_tensor[i]) * mask_value;
        sum += float_to_half(input_tensor[i + 1]) * mask_value;
        sum += float_to_half(input_tensor[i + 2]) * mask_value;
        sum += float_to_half(input_tensor[i + 3]) * mask_value;

        output[i / 4] = half_to_float(sum); 
    }
}

extern "C" {

void complex_tensor_operation(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_size = va_arg(args, int);

    // Extract target values tensor
    const float* target_values = va_arg(args, const float*);
    int target_values_size = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_target_values, *d_output;
    cudaMalloc(&d_input, input_tensor_size * sizeof(float));
    cudaMalloc(&d_target_values, target_values_size * sizeof(float));
    cudaMalloc(&d_output, (input_tensor_size / 4) * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_values, target_values, target_values_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (input_tensor_size + threadsPerBlock - 1) / threadsPerBlock;

    complex_tensor_operation_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_target_values, d_output, input_tensor_size, target_values_size
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, (input_tensor_size / 4) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target_values);
    cudaFree(d_output);
}

}  // extern "C"
