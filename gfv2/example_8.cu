
#include <cuda_runtime.h>
#include <cuda_fp16.h> // For half precision
#include <cublas_v2.h>  // For cuBLAS
#include <cudnn.h>      // For cuDNN

extern "C" {

void cholesky_decomposition(int num_args, ...) {
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

    // cuBLAS setup
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    // cuDNN setup
    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);

    // cuDNN Cholesky parameters
    cudnnTensorDescriptor_t xDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnSetTensorDescriptor(xDesc, CUDNN_DATA_FLOAT, 1, input_tensor_dim0, input_tensor_dim1, input_tensor_dim0, input_tensor_dim1);

    // Perform Cholesky decomposition using cuDNN
    cudnnStatus_t status = cudnnCholeskyForward(cudnnHandle,
                                             CUDNN_CHOLESKY_LOWER,
                                             xDesc,
                                             d_input,
                                             xDesc,
                                             d_output);

    if (status != CUDNN_STATUS_SUCCESS) {
        // Handle error
        printf("cuDNN Cholesky error: %d\n", status);
    }

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Destroy cuBLAS handle
    cublasDestroy(cublasHandle);

    // Destroy cuDNN handle
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroy(cudnnHandle);
}

}  // extern "C"
