
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cudnn.h>
#include <stdarg.h>

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

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);
    int weight_dim3 = va_arg(args, int);

    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    const int64_t* indices = va_arg(args, const int64_t*);
    int indices_dim0 = va_arg(args, int);
    int indices_dim1 = va_arg(args, int);
    int indices_dim2 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    // CUDNN setup
    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);

    cudnnTensorDescriptor_t inputDesc, weightDesc, biasDesc, outputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&weightDesc);
    cudnnCreateTensorDescriptor(&biasDesc);
    cudnnCreateTensorDescriptor(&outputDesc);

    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_tensor_dim0, input_tensor_dim1, input_tensor_dim2, input_tensor_dim3);
    cudnnSetTensor4dDescriptor(weightDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, weight_dim0, weight_dim1, weight_dim2, weight_dim3);
    cudnnSetTensor1dDescriptor(biasDesc, CUDNN_DATA_FLOAT, bias_dim0);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_tensor_dim0, weight_dim0, indices_dim1, indices_dim2);

    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION);

    // Allocate device memory
    float* d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * sizeof(float));
    cudaMalloc(&d_bias, bias_dim0 * sizeof(float));
    cudaMalloc(&d_output, input_tensor_dim0 * weight_dim0 * indices_dim1 * indices_dim2 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution using CUDNN
    cudnnConvolutionForward(cudnnHandle, 
                           1.0f, inputDesc, d_input, 
                           weightDesc, d_weight, 
                           convDesc, 
                           0, 0, 
                           1.0f, outputDesc, d_output);

    // Apply bias
    cudnnAddTensor(cudnnHandle, 
                  CUDNN_ADD_SAME_C, 
                  1.0f, biasDesc, d_bias, 
                  1.0f, outputDesc, d_output);

    // Perform min pooling
    float* d_min_output;
    cudaMalloc(&d_min_output, input_tensor_dim0 * weight_dim0 * indices_dim1 * indices_dim2 * sizeof(float));
    cudaMemcpy(d_min_output, d_output, input_tensor_dim0 * weight_dim0 * indices_dim1 * indices_dim2 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize(); // Synchronize before performing min pooling

    int min_axis = 1;
    int size = input_tensor_dim0 * weight_dim0 * indices_dim1 * indices_dim2;
    for (int i = 0; i < size; ++i) {
        float min_value = d_min_output[i];
        for (int j = i + input_tensor_dim0 * indices_dim1 * indices_dim2; j < size; j += input_tensor_dim0 * indices_dim1 * indices_dim2) {
            if (d_min_output[j] < min_value) {
                min_value = d_min_output[j];
            }
        }
        d_min_output[i] = min_value;
    }

    // Index select
    cudaMemcpy(d_output, d_min_output, input_tensor_dim0 * weight_dim0 * indices_dim1 * indices_dim2 * sizeof(float), cudaMemcpyDeviceToDevice);
    for (int i = 0; i < input_tensor_dim0; ++i) {
        for (int j = 0; j < indices_dim1; ++j) {
            for (int k = 0; k < indices_dim2; ++k) {
                int index = indices[i * indices_dim1 * indices_dim2 + j * indices_dim2 + k];
                d_output[i * weight_dim0 * indices_dim1 * indices_dim2 + j * indices_dim2 + k] = d_min_output[i * weight_dim0 * indices_dim1 * indices_dim2 + index * indices_dim2 + k];
            }
        }
    }

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor_dim0 * weight_dim0 * indices_dim1 * indices_dim2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaFree(d_min_output);

    // Cleanup CUDNN
    cudnnDestroy(inputDesc);
    cudnnDestroy(weightDesc);
    cudnnDestroy(biasDesc);
    cudnnDestroy(outputDesc);
    cudnnDestroy(convDesc);
    cudnnDestroy(cudnnHandle);
}

}
