
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end
#include <cublas_v2.h>

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

    // Extract temperature
    float temperature = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set up cuBLAS parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int m = batch_size;
    const int n = input_dim;

    // Launch cuBLAS softmax with temperature
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, n, n, &alpha, d_input, n, d_input, n, &beta, d_output, n, 1);

    // Apply temperature scaling
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            d_output[i * input_dim + j] /= temperature;
        }
    }

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Destroy cuBLAS handle
    cublasDestroy(handle);
}

}  // extern "C"
