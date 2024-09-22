
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHECK(condition)                                 \
    {                                                    \
        if (!(condition)) {                             \
            fprintf(stderr, "Error: " #condition "\n"); \
            exit(1);                                   \
        }                                                \
    }

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input1_data = va_arg(args, const float*);
    int input1_dim0 = va_arg(args, int);
    int input1_dim1 = va_arg(args, int);

    const float* input2_data = va_arg(args, const float*);
    int input2_dim0 = va_arg(args, int);
    int input2_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output_data = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input1, *d_input2, *d_output;
    CHECK(cudaMalloc(&d_input1, input1_dim0 * input1_dim1 * sizeof(float)));
    CHECK(cudaMalloc(&d_input2, input2_dim0 * input2_dim1 * sizeof(float)));
    CHECK(cudaMalloc(&d_output, input1_dim0 * input2_dim0 * sizeof(float)));

    // Copy input data to device
    CHECK(cudaMemcpy(d_input1, input1_data, input1_dim0 * input1_dim1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_input2, input2_data, input2_dim0 * input2_dim1 * sizeof(float), cudaMemcpyHostToDevice));

    // Use cuDNN for pairwise Manhattan distance calculation
    // Create cuDNN context
    cudnnHandle_t cudnnHandle;
    CHECK(cudnnCreate(&cudnnHandle));

    // Create cuDNN tensor descriptors
    cudnnTensorDescriptor_t input1Desc, input2Desc, outputDesc;
    CHECK(cudnnCreateTensorDescriptor(&input1Desc));
    CHECK(cudnnCreateTensorDescriptor(&input2Desc));
    CHECK(cudnnCreateTensorDescriptor(&outputDesc));

    // Set tensor descriptors
    CHECK(cudnnSetTensorNdDescriptor(input1Desc, CUDNN_DATA_FLOAT, 1, &input1_dim0));
    CHECK(cudnnSetTensorNdDescriptor(input2Desc, CUDNN_DATA_FLOAT, 1, &input2_dim0));
    CHECK(cudnnSetTensorNdDescriptor(outputDesc, CUDNN_DATA_FLOAT, 2, &input1_dim0, &input2_dim0));

    // Create cuDNN operation descriptor
    cudnnOpTensorDescriptor_t opDesc;
    CHECK(cudnnCreateOpTensorDescriptor(&opDesc));
    CHECK(cudnnSetOpTensorDescriptor(opDesc, CUDNN_OP_TENSOR_ADD,
                                     CUDNN_OP_TENSOR_MUL, CUDNN_PROPAGATE_NAN));

    // Perform pairwise Manhattan distance calculation using cuDNN
    // We use abs(input1 - input2).sum(dim=-1)
    // This can be achieved by:
    // 1. input1 - input2 (broadcasting)
    // 2. |input1 - input2|
    // 3. sum(dim=-1)
    // We can achieve 1 and 2 by using CUDNN_OP_TENSOR_ADD with
    // input1, -input2 and output as the first, second, and third
    // arguments. 
    // For 3, we can simply call cudnnTransformTensorEx directly on the
    // output. 

    // 1. Broadcast subtraction (input1 - input2)
    CHECK(cudnnTransformTensorEx(cudnnHandle, opDesc, input1Desc, d_input1,
                                   input2Desc, d_input2, 
                                   outputDesc, d_output));

    // 2. Take absolute value (|input1 - input2|)
    CHECK(cudnnTransformTensorEx(cudnnHandle, opDesc, outputDesc, d_output,
                                   outputDesc, d_output, 
                                   outputDesc, d_output));

    // 3. Sum along the last dimension (sum(dim=-1))
    CHECK(cudnnTransformTensorEx(cudnnHandle, opDesc, outputDesc, d_output,
                                   outputDesc, d_output, 
                                   outputDesc, d_output));

    // Copy result back to host
    CHECK(cudaMemcpy(output_data, d_output, input1_dim0 * input2_dim0 * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(d_input1));
    CHECK(cudaFree(d_input2));
    CHECK(cudaFree(d_output));

    // Free cuDNN resources
    CHECK(cudnnDestroyTensorDescriptor(input1Desc));
    CHECK(cudnnDestroyTensorDescriptor(input2Desc));
    CHECK(cudnnDestroyTensorDescriptor(outputDesc));
    CHECK(cudnnDestroyOpTensorDescriptor(opDesc));
    CHECK(cudnnDestroy(cudnnHandle));
}

}  // extern "C"
