
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end
#include <cufft.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
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

    // Extract signal_ndim
    int signal_ndim = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim1 = input_tensor_dim1;
    int input_dim2 = input_tensor_dim2;
    int output_dim2 = input_dim2 * 2; // Output dim2 is twice the input dim2

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim1 * input_dim2 * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim1 * output_dim2 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim1 * input_dim2 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform IRFFT using cuFFT
    cufftHandle plan;
    cufftPlanMany(&plan, 1, &input_dim1, &input_dim2, 1, &input_dim2, &batch_size, &output_dim2,
                   CUFFT_R2C, signal_ndim);

    cufftExecR2C(plan, d_input, d_output);
    cufftDestroy(plan);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_dim1 * output_dim2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"