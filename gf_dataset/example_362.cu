
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h> 
#include <cudnn.h>

#define CHECK(x)                                              \
  do                                                          \
  {                                                           \
    cudnnStatus_t status = (x);                               \
    if (status != CUDNN_STATUS_SUCCESS)                       \
    {                                                           \
      fprintf(stderr, "CUDNN error: %s:%d: %s\n", __FILE__,    \
              __LINE__, cudnnGetErrorString(status));          \
      exit(EXIT_FAILURE);                                     \
    }                                                           \
  } while (0)

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
    int bias_dim = va_arg(args, int);

    const int* target_tensor = va_arg(args, const int*);
    int target_tensor_dim0 = va_arg(args, int);
    int target_tensor_dim1 = va_arg(args, int);
    int target_tensor_dim2 = va_arg(args, int);
    int target_tensor_dim3 = va_arg(args, int);

    const float* sparsity_weight_ptr = va_arg(args, const float*);
    float sparsity_weight = *sparsity_weight_ptr;

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // CUDA Initialization
    cudnnHandle_t cudnn_handle;
    CHECK(cudnnCreate(&cudnn_handle));

    // Define tensor descriptors
    cudnnTensorDescriptor_t input_desc, weight_desc, output_desc, target_desc;
    CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CHECK(cudnnCreateTensorDescriptor(&weight_desc));
    CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CHECK(cudnnCreateTensorDescriptor(&target_desc));

    // Set tensor descriptors
    CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_DATA_FLOAT, 1, input_tensor_dim0, 
                                     input_tensor_dim1, input_tensor_dim2, input_tensor_dim3));
    CHECK(cudnnSetTensor4dDescriptor(weight_desc, CUDNN_DATA_FLOAT, 1, weight_dim0,
                                     weight_dim1, weight_dim2, weight_dim3));
    CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_DATA_FLOAT, 1, target_tensor_dim0,
                                     target_tensor_dim1, target_tensor_dim2, target_tensor_dim3));
    CHECK(cudnnSetTensor4dDescriptor(target_desc, CUDNN_DATA_INT32, 1, target_tensor_dim0,
                                     target_tensor_dim1, target_tensor_dim2, target_tensor_dim3));

    // Define convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION));

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    int *d_target;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * sizeof(float));
    cudaMalloc(&d_bias, bias_dim * sizeof(float));
    cudaMalloc(&d_output, target_tensor_dim0 * target_tensor_dim1 * target_tensor_dim2 * target_tensor_dim3 * sizeof(float));
    cudaMalloc(&d_target, target_tensor_dim0 * target_tensor_dim1 * target_tensor_dim2 * target_tensor_dim3 * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, target_tensor_dim0 * target_tensor_dim1 * target_tensor_dim2 * target_tensor_dim3 * sizeof(int), cudaMemcpyHostToDevice);

    // Perform convolution
    CHECK(cudnnConvolutionForward(cudnn_handle,
                                  &alpha, conv_desc, d_input, input_desc,
                                  d_weight, weight_desc,
                                  &beta, d_output, output_desc));

    // Add bias
    CHECK(cudnnAddTensor(cudnn_handle, 
                         &alpha, d_bias, input_desc,
                         &beta, d_output, output_desc));

    // Calculate cross-entropy loss with cudnn
    cudnnSoftmaxDescriptor_t softmax_desc;
    CHECK(cudnnCreateSoftmaxDescriptor(&softmax_desc));
    CHECK(cudnnSetSoftmaxDescriptor(softmax_desc, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE));

    float loss;
    CHECK(cudnnSoftmaxForward(cudnn_handle, softmax_desc, d_output, output_desc, &loss));

    // Calculate KL divergence loss
    float kl_div_loss;
    cudaMalloc(&kl_div_loss, sizeof(float));
    CHECK(cudaLaunchKernel(
        // Kernel function
        (const void *) &kl_div_loss_kernel,
        // Threads per block
        dim3(1, 1, 1),
        // Blocks per grid
        dim3(1, 1, 1),
        // Shared memory
        0,
        // Streams
        0,
        // Kernel arguments
        d_weight, weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3,
        &kl_div_loss,
        &sparsity_weight
    ));

    // Copy result back to host
    cudaMemcpy(output, d_output, target_tensor_dim0 * target_tensor_dim1 * target_tensor_dim2 * target_tensor_dim3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaFree(d_target);

    // Free CUDA descriptors
    CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CHECK(cudnnDestroyTensorDescriptor(weight_desc));
    CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CHECK(cudnnDestroyTensorDescriptor(target_desc));
    CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK(cudnnDestroySoftmaxDescriptor(softmax_desc));

    // Destroy CUDA handle
    CHECK(cudnnDestroy(cudnn_handle));
}

__global__ void kl_div_loss_kernel(const float* weight, int weight_size, float* kl_div_loss, float sparsity_weight) {
    float sum = 0.0f;
    for (int i = 0; i < weight_size; ++i) {
        float val = weight[i];
        sum += sparsity_weight * (logf(fabsf(val) + 1e-6f) - val);
    }
    *kl_div_loss = sum;
}

}  // extern "C"
