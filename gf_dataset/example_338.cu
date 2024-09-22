
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For half precision
#include <cudnn.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
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
    int input_tensor_dim3 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);
    int weight_dim3 = va_arg(args, int);

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Cudnn setup
    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);

    cudnnTensorDescriptor_t inputDesc, weightDesc, biasDesc, outputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&weightDesc);
    cudnnCreateTensorDescriptor(&biasDesc);
    cudnnCreateTensorDescriptor(&outputDesc);

    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_tensor_dim0, input_tensor_dim1, 
                                input_tensor_dim2, input_tensor_dim3);
    cudnnSetTensor4dDescriptor(weightDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, weight_dim0, weight_dim1,
                                weight_dim2, weight_dim3);
    cudnnSetTensor1dDescriptor(biasDesc, CUDNN_DATA_FLOAT, bias_dim0);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_tensor_dim0, input_tensor_dim1,
                                input_tensor_dim2, input_tensor_dim3);

    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolutionNdDescriptor(convDesc, 2, // Number of dimensions
                                  {1, 1}, // Padding in each dimension
                                  {0, 0}, // Stride in each dimension
                                  CUDNN_CONVOLUTION_CROSS_CORRELATION, CUDNN_DATA_FLOAT); //  

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * sizeof(float));
    cudaMalloc(&d_bias, bias_dim0 * sizeof(float));
    cudaMalloc(&d_output, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution with cuDNN
    cudnnConvolutionForward(cudnnHandle, 
                           &one, convDesc, 
                           inputDesc, d_input, 
                           weightDesc, d_weight, 
                           &zero, outputDesc, d_output);

    // Apply ReLU
    cudnnActivationDescriptor_t reluDesc;
    cudnnCreateActivationDescriptor(&reluDesc);
    cudnnSetActivationDescriptor(reluDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0f);
    cudnnActivationForward(cudnnHandle, reluDesc, outputDesc, d_output, outputDesc, d_output);

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up cuDNN resources
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(weightDesc);
    cudnnDestroyTensorDescriptor(biasDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyActivationDescriptor(reluDesc);
    cudnnDestroy(cudnnHandle);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
