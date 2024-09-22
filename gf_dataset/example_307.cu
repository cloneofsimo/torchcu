
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end
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

void learned_positional_encoding_bf16_max_filter(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract learned positional encoding tensor
    const float* learned_positional_encoding = va_arg(args, const float*);
    int learned_positional_encoding_dim0 = va_arg(args, int);
    int learned_positional_encoding_dim1 = va_arg(args, int);
    int learned_positional_encoding_dim2 = va_arg(args, int);
    int learned_positional_encoding_dim3 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // CUDA context creation and error checking
    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);

    // Allocate device memory
    float *d_input, *d_learned_positional_encoding, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float));
    cudaMalloc(&d_learned_positional_encoding, learned_positional_encoding_dim0 * learned_positional_encoding_dim1 * learned_positional_encoding_dim2 * learned_positional_encoding_dim3 * sizeof(float));
    cudaMalloc(&d_output, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_learned_positional_encoding, learned_positional_encoding, learned_positional_encoding_dim0 * learned_positional_encoding_dim1 * learned_positional_encoding_dim2 * learned_positional_encoding_dim3 * sizeof(float), cudaMemcpyHostToDevice);

    // Create cudnn tensor descriptors
    cudnnTensorDescriptor_t input_tensor_desc, learned_positional_encoding_desc, output_desc;
    cudnnCreateTensorDescriptor(&input_tensor_desc);
    cudnnCreateTensorDescriptor(&learned_positional_encoding_desc);
    cudnnCreateTensorDescriptor(&output_desc);

    // Set tensor descriptor dimensions (NHWC)
    cudnnSetTensor4dDescriptor(input_tensor_desc, CUDNN_DATA_FLOAT, 1, input_tensor_dim0, input_tensor_dim1, input_tensor_dim2, input_tensor_dim3);
    cudnnSetTensor4dDescriptor(learned_positional_encoding_desc, CUDNN_DATA_FLOAT, 1, learned_positional_encoding_dim0, learned_positional_encoding_dim1, learned_positional_encoding_dim2, learned_positional_encoding_dim3);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_DATA_FLOAT, 1, input_tensor_dim0, input_tensor_dim1, input_tensor_dim2, input_tensor_dim3);

    // Create cudnn filter descriptor for max pooling
    cudnnFilterDescriptor_t max_pool_filter_desc;
    cudnnCreateFilterDescriptor(&max_pool_filter_desc);
    cudnnSetFilter4dDescriptor(max_pool_filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_FORMAT_NHWC, 3, 3, 1, 1);

    // Create cudnn pooling descriptor
    cudnnPoolingDescriptor_t max_pool_desc;
    cudnnCreatePoolingDescriptor(&max_pool_desc);
    cudnnSetPoolingNdDescriptor(max_pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 3, 3, 1, 1, 1, 1, 1, 1);

    // Perform the learned positional encoding addition on the device
    cudnnAddTensor(cudnnHandle, CUDNN_ADD_SAME_C, 
                  &input_tensor_desc, d_input,
                  &learned_positional_encoding_desc, d_learned_positional_encoding,
                  &output_desc, d_output);

    // Perform max pooling with bfloat16 conversion using cudnn
    cudnnDataType_t data_type = CUDNN_DATA_BFLOAT16;
    cudnnSetTensorDescriptorEx(input_tensor_desc, data_type, 1, input_tensor_dim0, input_tensor_dim1, input_tensor_dim2, input_tensor_dim3);
    cudnnSetTensorDescriptorEx(output_desc, data_type, 1, input_tensor_dim0, input_tensor_dim1, input_tensor_dim2, input_tensor_dim3);

    cudnnPoolingForward(cudnnHandle, max_pool_desc, 
                        input_tensor_desc, d_output,
                        output_desc, d_output);
                        
    // Convert back to float32
    cudnnSetTensorDescriptorEx(output_desc, CUDNN_DATA_FLOAT, 1, input_tensor_dim0, input_tensor_dim1, input_tensor_dim2, input_tensor_dim3);

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory and cudnn descriptors
    cudaFree(d_input);
    cudaFree(d_learned_positional_encoding);
    cudaFree(d_output);
    cudnnDestroyTensorDescriptor(input_tensor_desc);
    cudnnDestroyTensorDescriptor(learned_positional_encoding_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(max_pool_filter_desc);
    cudnnDestroyPoolingDescriptor(max_pool_desc);
    cudnnDestroy(cudnnHandle);
}

}  // extern "C"
