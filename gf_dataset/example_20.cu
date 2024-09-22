
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f); 
}

// Helper function to convert half to int8
__device__ __forceinline__ char half_to_int8(half h) {
    return __int_as_char(static_cast<int>(h)); 
}

__global__ void pitch_correction_flatten_int8_kernel(const float* input_tensor, float pitch_shift, float scale, 
                                                        char* output, int input_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < input_size) {
        // Pitch correction
        int corrected_index = static_cast<int>(i * pitch_shift);
        if (corrected_index >= 0 && corrected_index < input_size) {
            // Convert to half (fp16) for intermediate calculations
            half input_value = float_to_half(input_tensor[corrected_index]);
            // Scale and quantize to int8
            output[i] = half_to_int8(input_value * scale);
        } else {
            output[i] = 0; // Handle out-of-bounds indices
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

    // Extract pitch shift
    float pitch_shift = va_arg(args, float);
    
    // Extract scale
    float scale = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    char* output = va_arg(args, char*);

    va_end(args);

    int input_size = input_tensor_dim0 * input_tensor_dim1;

    // Allocate device memory
    float *d_input;
    cudaMalloc(&d_input, input_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (input_size + threadsPerBlock - 1) / threadsPerBlock;

    pitch_correction_flatten_int8_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, pitch_shift, scale, output, input_size
    );

    // Free device memory
    cudaFree(d_input);
}

}  // extern "C"
