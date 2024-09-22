
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function to convert float to __half
__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert __half to float
__device__ __forceinline__ float half_to_float(__half h) {
    return __half2float(h);
}

__global__ void idft_normalize_flatten_int8_fp16_kernel(const cufftComplex* input, float scale, 
                                                         int8_t* output, int batch_size, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * input_size) {
        int batch_idx = idx / input_size;
        int channel_idx = idx % input_size;

        // Perform inverse FFT
        cufftComplex result;
        cufftExecC2R(result, input + batch_idx * input_size + channel_idx, 1);

        // Normalize
        float normalized = result.x / scale;

        // Quantize to int8
        output[idx] = __int_as_char(normalized);
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
    int input_tensor_dim2 = va_arg(args, int);

    // Extract scale
    const float* scale_ptr = va_arg(args, const float*);
    float scale = *scale_ptr;

    // Extract output tensor (assuming it's preallocated)
    int8_t* output = va_arg(args, int8_t*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_size = input_tensor_dim1 * input_tensor_dim2;

    // Allocate device memory
    cufftComplex* d_input;
    int8_t* d_output;
    cudaMalloc(&d_input, batch_size * input_size * sizeof(cufftComplex));
    cudaMalloc(&d_output, batch_size * input_size * sizeof(int8_t));

    // Copy input data to device (convert float to cufftComplex)
    for (int i = 0; i < batch_size * input_size; ++i) {
        d_input[i].x = input_tensor[2 * i];
        d_input[i].y = input_tensor[2 * i + 1];
    }

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size * input_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    idft_normalize_flatten_int8_fp16_kernel<<<numBlocks, threadsPerBlock>>>(d_input, scale, d_output, 
                                                                         batch_size, input_size);

    // Copy result back to host (convert int8 to float16)
    cudaMemcpy(output, d_output, batch_size * input_size * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
