
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

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

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
    cudaMalloc(&d_output, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Use cuBLAS for Cholesky decomposition
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Convert input to bfloat16 on device
    float *d_input_bf16;
    cudaMalloc(&d_input_bf16, input_tensor_dim0 * input_tensor_dim1 * sizeof(__nv_bfloat16));
    cudaMemcpy(d_input_bf16, d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyDeviceToDevice);

    // Perform Cholesky decomposition
    cublasPointerMode_t pointerMode;
    cublasGetPointerMode(handle, &pointerMode);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE); 
    cublasCholesky(handle, CUBLAS_FILL_MODE_UPPER, input_tensor_dim0, d_input_bf16, 
                    input_tensor_dim0, 1.0f);

    // Convert result back to float on device
    cudaMemcpy(d_output, d_input_bf16, input_tensor_dim0 * input_tensor_dim1 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input_bf16);
}

}  // extern "C"
