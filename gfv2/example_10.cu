
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <iostream>

extern "C" {

void torch_sobel_filter_int8_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    // Extract output tensor
    int8_t* output = va_arg(args, int8_t*);

    va_end(args);

    // CUDA setup
    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);

    // Input and output tensor descriptors
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);

    // Set input tensor descriptor
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_DATA_FLOAT, 1, input_tensor_dim1, input_tensor_dim2, input_tensor_dim3);
    // Set output tensor descriptor (int8)
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_DATA_INT8, 1, input_tensor_dim1, input_tensor_dim2, input_tensor_dim3);

    // Sobel kernel descriptor
    cudnnFilterDescriptor_t sobelFilterDesc;
    cudnnCreateFilterDescriptor(&sobelFilterDesc);
    const float sobelX[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    const float sobelY[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    cudnnSetFilterNdDescriptor(sobelFilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 3, 3); // 1x1x3x3
    cudnnSetFilterNdDescriptor(sobelFilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 3, 3); // 1x1x3x3
    cudaMemcpy(cudnnFilterDesc, sobelX, sizeof(sobelX), cudaMemcpyHostToDevice);
    cudaMemcpy(cudnnFilterDesc, sobelY, sizeof(sobelY), cudaMemcpyHostToDevice);

    // Convolution descriptor
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolutionNdDescriptor(convDesc, 0, 1, 1, 1, 1, 1, CUDNN_CROSS_CHANNEL_PRODUCT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT);

    // Allocate device memory for input, output, and gradient
    float* d_input;
    int8_t* d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float));
    cudaMalloc(&d_output, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(int8_t));

    // Copy input to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float), cudaMemcpyHostToDevice);

    // Set up convolution parameters
    cudnnConvolutionFwdAlgo_t convolutionAlgorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM; // Optimize for GEMM
    int workspaceSize;
    cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDesc, sobelFilterDesc, convDesc, outputDesc, convolutionAlgorithm, &workspaceSize);
    char* workspace = new char[workspaceSize];

    // Perform convolution (forward pass)
    cudnnConvolutionForward(cudnnHandle,
        1.0f, // alpha
        inputDesc, d_input,
        sobelFilterDesc, nullptr, // No bias
        convDesc,
        convolutionAlgorithm, workspace, workspaceSize,
        0.0f, // beta
        outputDesc, d_output);

    // Copy output back to host
    cudaMemcpy(output, d_output, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Cleanup CUDA resources
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(sobelFilterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnnHandle);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] workspace;
}

}  // extern "C"
