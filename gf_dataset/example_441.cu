
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cudnn.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

extern "C" {

void time_stretch_orthogonal_regularization_bf16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int seq_len = va_arg(args, int);
    int features = va_arg(args, int);

    // Extract stretch_factor
    float stretch_factor = va_arg(args, float);

    // Extract regularization_weight
    float regularization_weight = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_stretched_output;
    cudaMalloc(&d_input, batch_size * seq_len * features * sizeof(float));
    cudaMalloc(&d_stretched_output, batch_size * int(seq_len * stretch_factor) * features * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * seq_len * features * sizeof(float), cudaMemcpyHostToDevice);

    // Time stretching using cuDNN
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);
    
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);

    cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 3,  // CUDNN_DATA_BFLOAT16 for bfloat16
                                 const_cast<int*>(reinterpret_cast<const int*>(&batch_size)),
                                 const_cast<int*>(reinterpret_cast<const int*>(&seq_len)),
                                 const_cast<int*>(reinterpret_cast<const int*>(&features)));
    cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, 3,  // CUDNN_DATA_BFLOAT16 for bfloat16
                                 const_cast<int*>(reinterpret_cast<const int*>(&batch_size)),
                                 const_cast<int*>(reinterpret_cast<const int*>(&int(seq_len * stretch_factor))),
                                 const_cast<int*>(reinterpret_cast<const int*>(&features)));

    cudnnReshapeTensorDescriptor(output_desc, 
                                  const_cast<int*>(reinterpret_cast<const int*>(&batch_size)),
                                  const_cast<int*>(reinterpret_cast<const int*>(&int(seq_len * stretch_factor))),
                                  const_cast<int*>(reinterpret_cast<const int*>(&features)));

    cudnnTensorDescriptor_t stretch_descriptor;
    cudnnCreateTensorDescriptor(&stretch_descriptor);
    cudnnSetTensorNdDescriptor(stretch_descriptor, CUDNN_DATA_FLOAT, 1, // CUDNN_DATA_BFLOAT16 for bfloat16
                                 const_cast<int*>(reinterpret_cast<const int*>(&stretch_factor)));

    cudnnDataType_t data_type = CUDNN_DATA_FLOAT; // CUDNN_DATA_BFLOAT16 for bfloat16
    cudnnInterpolationMode_t mode = CUDNN_INTERPOLATION_LINEAR;

    cudnnPerformTensorOp(cudnn_handle, CUDNN_OP_TENSOR_INTERPOLATION,
                         data_type, input_desc, d_input,
                         data_type, output_desc, d_stretched_output,
                         stretch_descriptor, mode);

    // Orthogonal regularization using cuBLAS
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    float *d_orthogonal_loss = (float*)malloc(sizeof(float));
    cudaMalloc(&d_orthogonal_loss, sizeof(float));

    // Calculate orthogonal loss for each time step
    for (int i = 0; i < int(seq_len * stretch_factor); i++) {
        int W_size = features;
        int W_stride = int(seq_len * stretch_factor) * features;
        int W_offset = i * features;

        // Calculate W^T * W
        float* d_WtW = (float*)malloc(features * features * sizeof(float));
        cudaMalloc(&d_WtW, features * features * sizeof(float));
        cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    features, features, features,
                    &one, d_stretched_output + W_offset, W_stride,
                    d_stretched_output + W_offset, W_stride,
                    &zero, d_WtW, features);

        // Calculate ||W^T * W - I||
        float* d_I = (float*)malloc(features * features * sizeof(float));
        cudaMalloc(&d_I, features * features * sizeof(float));
        for (int k = 0; k < features; k++) {
            d_I[k * features + k] = 1.0f;
        }

        float* d_WtW_minus_I = (float*)malloc(features * features * sizeof(float));
        cudaMalloc(&d_WtW_minus_I, features * features * sizeof(float));
        cublasSaxpy(cublas_handle, features * features, -1.0f, d_I, 1, d_WtW, 1);
        cublasSnrm2(cublas_handle, features * features, d_WtW_minus_I, 1, d_orthogonal_loss);

        // Add orthogonal loss to corresponding time step
        float *d_orthogonal_loss_value = (float*)malloc(sizeof(float));
        cudaMalloc(&d_orthogonal_loss_value, sizeof(float));
        cudaMemcpy(d_orthogonal_loss_value, d_orthogonal_loss, sizeof(float), cudaMemcpyDeviceToDevice);
        cublasSaxpy(cublas_handle, features, regularization_weight, d_orthogonal_loss_value, 1, d_stretched_output + W_offset, W_stride);

        // Free temporary memory
        cudaFree(d_WtW);
        cudaFree(d_I);
        cudaFree(d_WtW_minus_I);
        cudaFree(d_orthogonal_loss_value);
    }

    // Copy result back to host
    cudaMemcpy(output, d_stretched_output, batch_size * int(seq_len * stretch_factor) * features * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_stretched_output);
    cudaFree(d_orthogonal_loss);

    // Destroy cuDNN and cuBLAS handles
    cudnnDestroy(input_desc);
    cudnnDestroy(output_desc);
    cudnnDestroy(stretch_descriptor);
    cudnnDestroy(cudnn_handle);

    cublasDestroy(cublas_handle);

    // Free temporary host memory
    free(d_orthogonal_loss);
}
} // extern "C"
