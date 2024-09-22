
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// Helper function for int8 conversion
__device__ __forceinline__ int8_t float_to_int8(float f) {
    return static_cast<int8_t>(f);
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input1 = va_arg(args, const float*);
    int input1_dim0 = va_arg(args, int);
    int input1_dim1 = va_arg(args, int);
    int input1_dim2 = va_arg(args, int);

    const float* input2 = va_arg(args, const float*);
    int input2_dim0 = va_arg(args, int);
    int input2_dim1 = va_arg(args, int);
    int input2_dim2 = va_arg(args, int);

    const float* input3 = va_arg(args, const float*);
    int input3_dim0 = va_arg(args, int);
    int input3_dim1 = va_arg(args, int);
    int input3_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    int8_t* output = va_arg(args, int8_t*);

    va_end(args);

    // Allocate device memory
    float *d_input1, *d_input2, *d_input3;
    int8_t *d_output;
    cudaMalloc(&d_input1, input1_dim0 * input1_dim1 * input1_dim2 * sizeof(float));
    cudaMalloc(&d_input2, input2_dim0 * input2_dim1 * input2_dim2 * sizeof(float));
    cudaMalloc(&d_input3, input3_dim0 * input3_dim1 * input3_dim2 * sizeof(float));
    cudaMalloc(&d_output, input1_dim0 * 1 * input1_dim1 * input2_dim2 * sizeof(int8_t));

    // Copy input data to device
    cudaMemcpy(d_input1, input1, input1_dim0 * input1_dim1 * input1_dim2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2, input2_dim0 * input2_dim1 * input2_dim2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input3, input3, input3_dim0 * input3_dim1 * input3_dim2 * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform bmm on the device using cuBLAS
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, input1_dim1, input2_dim2, input1_dim2,
                &alpha, d_input1, input1_dim1, d_input2, input2_dim2,
                &beta, d_output, input1_dim1);

    // Unsqueeze the result
    // (Since CUDA does not directly support unsqueeze, we can manually do it in the loop below)

    // Copy output data back to host with int8 conversion
    for (int i = 0; i < input1_dim0; ++i) {
        for (int j = 0; j < input1_dim1; ++j) {
            for (int k = 0; k < input2_dim2; ++k) {
                output[i * input1_dim1 * input2_dim2 + j * input2_dim2 + k] =
                    float_to_int8(d_output[i * input1_dim1 * input2_dim2 + j * input2_dim2 + k]);
            }
        }
    }

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_input3);
    cudaFree(d_output);

    // Destroy cuBLAS handle
    cublasDestroy(handle);
}

}  // extern "C"
