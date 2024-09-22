
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

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

    // Extract bias tensor
    const float* bias = va_arg(args, const float*);
    int bias_dim = va_arg(args, int);

    // Extract dropout probability
    float p = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_bias, output_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize cuDNN
    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);

    // Create cuDNN tensors
    cudnnTensorDescriptor_t inputDesc, weightDesc, biasDesc, outputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&weightDesc);
    cudnnCreateTensorDescriptor(&biasDesc);
    cudnnCreateTensorDescriptor(&outputDesc);

    // Set tensor descriptions
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, input_dim, 1);
    cudnnSetTensor4dDescriptor(weightDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_dim, input_dim, 1);
    cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_dim, 1, 1);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, output_dim, 1);

    // Create cuDNN dropout descriptor
    cudnnDropoutDescriptor_t dropoutDesc;
    cudnnCreateDropoutDescriptor(&dropoutDesc);
    cudnnSetDropoutDescriptor(dropoutDesc, p, 0, 0, 0);

    // Create cuDNN activation descriptor
    cudnnActivationDescriptor_t activationDesc;
    cudnnCreateActivationDescriptor(&activationDesc);
    cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0f);

    // Perform dropout
    cudnnDropoutForward(cudnnHandle, dropoutDesc, d_input, inputDesc, d_input, inputDesc);

    // Perform linear transformation
    cudnnAddTensor(cudnnHandle, CUDNN_ADD_SAME, &one, biasDesc, d_bias, &one, outputDesc, d_output, CUDNN_DATA_FLOAT, 0);
    cudnnConvolutionForward(cudnnHandle, 
                             &one, 
                             inputDesc, d_input, 
                             weightDesc, d_weight, 
                             biasDesc, d_bias, 
                             0, cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputDesc, weightDesc, outputDesc, 0, 0), 0, 
                             outputDesc, d_output); 

    // Perform ReLU activation
    cudnnActivationForward(cudnnHandle, activationDesc, outputDesc, d_output, outputDesc, d_output);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up cuDNN
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(weightDesc);
    cudnnDestroyTensorDescriptor(biasDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyDropoutDescriptor(dropoutDesc);
    cudnnDestroyActivationDescriptor(activationDesc);
    cudnnDestroy(cudnnHandle);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

}  // extern "C"
