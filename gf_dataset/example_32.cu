
#include <cuda_runtime.h>
#include <cufft.h>  // Include cuFFT library
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);  // Use round-to-nearest mode for conversion
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    half* output = va_arg(args, half*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize cuFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, input_dim, CUFFT_C2C, batch_size);  // Assuming complex-to-complex FFT

    // Perform FFT
    cufftExecC2C(plan, d_input, d_input, CUFFT_FORWARD);  // In-place FFT

    // Convert result to half-precision and copy back to host
    for (int i = 0; i < batch_size * input_dim; ++i) {
        output[i] = float_to_half(d_input[i]);
    }

    // Free device memory
    cudaFree(d_input);
    cufftDestroy(plan);
}

}  // extern "C"
