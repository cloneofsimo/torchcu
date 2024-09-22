
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

// Function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

extern "C" {

void image_jacobian_function(int num_args, ...) {
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

    // --- CUDNN setup ---
    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);

    // --- Create CUDNN tensors ---
    cudnnTensorDescriptor_t inputTensorDesc, weightTensorDesc, outputTensorDesc;
    cudnnCreateTensorDescriptor(&inputTensorDesc);
    cudnnCreateTensorDescriptor(&weightTensorDesc);
    cudnnCreateTensorDescriptor(&outputTensorDesc);

    // --- Set tensor descriptors ---
    cudnnSetTensorNdDescriptor(inputTensorDesc, CUDNN_DATA_FLOAT, 4, 
                                &input_tensor_dim0, &input_tensor_dim1, 
                                &input_tensor_dim2, &input_tensor_dim3);
    cudnnSetTensorNdDescriptor(weightTensorDesc, CUDNN_DATA_FLOAT, 4,
                                &weight_dim0, &weight_dim1, 
                                &weight_dim2, &weight_dim3);
    cudnnSetTensorNdDescriptor(outputTensorDesc, CUDNN_DATA_FLOAT, 4, 
                                &input_tensor_dim0, &weight_dim0,
                                &input_tensor_dim2, &input_tensor_dim3);

    // --- Create convolution descriptor ---
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolutionNdDescriptor(convDesc, 2, // Number of spatial dimensions
                                    {weight_dim2, weight_dim3}, 
                                    {0, 0}, {0, 0}, 
                                    CUDNN_CONVOLUTION_CROSS_CORRELATION, 
                                    CUDNN_DATA_FLOAT); // Data type

    // --- Allocate device memory ---
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * sizeof(float));
    cudaMalloc(&d_bias, bias_dim0 * sizeof(float));
    cudaMalloc(&d_output, input_tensor_dim0 * weight_dim0 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float));

    // --- Copy data to device ---
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // --- Perform convolution with CUDNN ---
    cudnnConvolutionForward(cudnnHandle, // CUDNN handle
                            &alpha, // Alpha value for scaling output
                            inputTensorDesc, // Input tensor descriptor
                            d_input, // Input tensor data
                            weightTensorDesc, // Weight tensor descriptor
                            d_weight, // Weight tensor data
                            convDesc, // Convolution descriptor
                            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, // Algorithm
                            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, // Preference for fastest algorithm
                            &beta, // Beta value for scaling output
                            outputTensorDesc, // Output tensor descriptor
                            d_output // Output tensor data
                            );

    // --- Apply bias ---
    cudnnAddTensor(cudnnHandle, // CUDNN handle
                   &alpha, // Alpha value for scaling bias
                   biasTensorDesc, // Bias tensor descriptor
                   d_bias, // Bias tensor data
                   &beta, // Beta value for scaling output
                   outputTensorDesc, // Output tensor descriptor
                   d_output // Output tensor data
                   );

    // --- Copy result back to host ---
    cudaMemcpy(output, d_output, input_tensor_dim0 * weight_dim0 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float), cudaMemcpyDeviceToHost);

    // --- Free CUDNN resources ---
    cudnnDestroyTensorDescriptor(inputTensorDesc);
    cudnnDestroyTensorDescriptor(weightTensorDesc);
    cudnnDestroyTensorDescriptor(outputTensorDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnnHandle);

    // --- Free device memory ---
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}
} // extern "C"
