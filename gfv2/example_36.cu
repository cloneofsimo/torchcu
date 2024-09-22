
#include <cuda_runtime.h>
#include <cudnn.h>

extern "C" {

void elementwise_min_cudnn(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor1 = va_arg(args, const float*);
    int input_tensor1_dim0 = va_arg(args, int);
    int input_tensor1_dim1 = va_arg(args, int);
    const float* input_tensor2 = va_arg(args, const float*);
    int input_tensor2_dim0 = va_arg(args, int);
    int input_tensor2_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Check input tensor shapes
    if (input_tensor1_dim0 != input_tensor2_dim0 ||
        input_tensor1_dim1 != input_tensor2_dim1) {
        printf("Error: Input tensors have incompatible shapes.\n");
        return;
    }

    // Initialize cuDNN
    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);

    // Allocate device memory
    float* d_input1, *d_input2, *d_output;
    cudaMalloc(&d_input1, input_tensor1_dim0 * input_tensor1_dim1 * sizeof(float));
    cudaMalloc(&d_input2, input_tensor2_dim0 * input_tensor2_dim1 * sizeof(float));
    cudaMalloc(&d_output, input_tensor1_dim0 * input_tensor1_dim1 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input1, input_tensor1, input_tensor1_dim0 * input_tensor1_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input_tensor2, input_tensor2_dim0 * input_tensor2_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuDNN tensors
    cudnnTensorDescriptor_t input_tensor_desc1, input_tensor_desc2, output_tensor_desc;
    cudnnCreateTensorDescriptor(&input_tensor_desc1);
    cudnnCreateTensorDescriptor(&input_tensor_desc2);
    cudnnCreateTensorDescriptor(&output_tensor_desc);

    // Set tensor descriptors
    cudnnSetTensor4dDescriptor(input_tensor_desc1, CUDNN_DATA_FLOAT, 1, 1, input_tensor1_dim0, input_tensor1_dim1);
    cudnnSetTensor4dDescriptor(input_tensor_desc2, CUDNN_DATA_FLOAT, 1, 1, input_tensor2_dim0, input_tensor2_dim1);
    cudnnSetTensor4dDescriptor(output_tensor_desc, CUDNN_DATA_FLOAT, 1, 1, input_tensor1_dim0, input_tensor1_dim1);

    // cuDNN operation
    cudnnActivationDescriptor_t actDesc;
    cudnnCreateActivationDescriptor(&actDesc);
    cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_MIN, CUDNN_PROPAGATE_NAN, 0.0f);

    cudnnActivationForward(cudnnHandle, actDesc, d_input1, input_tensor_desc1, d_input2, input_tensor_desc2, d_output, output_tensor_desc);

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor1_dim0 * input_tensor1_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudnnDestroyTensorDescriptor(input_tensor_desc1);
    cudnnDestroyTensorDescriptor(input_tensor_desc2);
    cudnnDestroyTensorDescriptor(output_tensor_desc);
    cudnnDestroyActivationDescriptor(actDesc);
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
    cudnnDestroy(cudnnHandle);
}

} // extern "C"
