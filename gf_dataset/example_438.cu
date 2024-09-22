
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * sizeof(float));
    cudaMalloc(&d_output, input_tensor_dim0 * weight_dim0 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform in-place square root on the device
    cudaSqrt(d_input, d_input, input_tensor_dim0 * input_tensor_dim1);

    // Use cuBLAS for einsum operation (transposing weight on the fly)
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMatrix(input_tensor_dim0, input_tensor_dim1, sizeof(float), d_input, input_tensor_dim1, d_input, input_tensor_dim1);
    cublasSetMatrix(weight_dim1, weight_dim0, sizeof(float), d_weight, weight_dim1, d_weight, weight_dim1);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, input_tensor_dim0, weight_dim0, input_tensor_dim1,
               &alpha, d_input, input_tensor_dim1, d_weight, weight_dim1, &beta, d_output, weight_dim0);

    cublasDestroy(handle);

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor_dim0 * weight_dim0 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}
