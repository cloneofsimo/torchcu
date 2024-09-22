
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>

// CUDA kernel for sigmoid backward using cuDNN
extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    const float* grad_output = va_arg(args, const float*);
    int grad_output_dim0 = va_arg(args, int);
    int grad_output_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* grad_input = va_arg(args, float*);

    va_end(args);

    // cuDNN setup
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t input_desc, grad_output_desc, grad_input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&grad_output_desc);
    cudnnCreateTensorDescriptor(&grad_input_desc);

    // Define tensor dimensions
    const int dims[] = {input_tensor_dim0, input_tensor_dim1};
    cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 2, dims);
    cudnnSetTensorNdDescriptor(grad_output_desc, CUDNN_DATA_FLOAT, 2, dims);
    cudnnSetTensorNdDescriptor(grad_input_desc, CUDNN_DATA_FLOAT, 2, dims);

    // Allocate device memory
    float *d_input, *d_grad_output, *d_grad_input;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
    cudaMalloc(&d_grad_output, grad_output_dim0 * grad_output_dim1 * sizeof(float));
    cudaMalloc(&d_grad_input, grad_input_dim0 * grad_input_dim1 * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, grad_output, grad_output_dim0 * grad_output_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform sigmoid backward using cuDNN
    cudnnActivationDescriptor_t act_desc;
    cudnnCreateActivationDescriptor(&act_desc);
    cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0f);

    cudnnActivationBackward(handle, act_desc, CUDNN_ACTIVATION_SIGMOID_BACKWARD,
                             d_input, input_desc, d_grad_output, grad_output_desc,
                             d_grad_input, grad_input_desc);

    // Copy result back to host
    cudaMemcpy(grad_input, d_grad_input, grad_input_dim0 * grad_input_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free resources
    cudaFree(d_input);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);

    cudnnDestroyActivationDescriptor(act_desc);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(grad_output_desc);
    cudnnDestroyTensorDescriptor(grad_input_desc);
    cudnnDestroy(handle);
}

}  // extern "C"
