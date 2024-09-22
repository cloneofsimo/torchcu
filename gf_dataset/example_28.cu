
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For half precision support
#include <device_launch_parameters.h>
#include <stdarg.h>

#define CUDA_CHECK(x) { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(-1); } }

// CUDA kernel for padding and diagonal element extraction
__global__ void pad_and_extract_diagonal(const float* input, float* padded_input, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        padded_input[row * n + col] = input[row * m + col];
    } else if (row < m && col >= m) {
        padded_input[row * n + col] = 0.0f;
    }
}

// CUDA kernel for log_softmax (using cudnn)
__global__ void log_softmax_kernel(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = logf(expf(input[i]) / (expf(input[i]) + expf(input[i + 1])));
    }
}

// CUDA kernel for multi-label margin loss
__global__ void multilabel_margin_loss_kernel(const float* log_softmax_probs, const float* diagonal_elements, float* loss, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        loss[0] += fmaxf(0.0f, log_softmax_probs[i] - diagonal_elements[i]);
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract target tensor
    const int* target_tensor = va_arg(args, const int*);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Calculate padded size
    int padded_size = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_padded_input, *d_log_softmax_probs, *d_loss;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
    cudaMalloc(&d_padded_input, input_tensor_dim0 * padded_size * sizeof(float));
    cudaMalloc(&d_log_softmax_probs, input_tensor_dim0 * padded_size * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch padding kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((padded_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_tensor_dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y);
    pad_and_extract_diagonal<<<numBlocks, threadsPerBlock>>>(d_input, d_padded_input, input_tensor_dim0, padded_size);

    // Compute log_softmax using cudnn
    cudnnHandle_t cudnnHandle;
    CUDA_CHECK(cudnnCreate(&cudnnHandle));
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    CUDA_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDA_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    CUDA_CHECK(cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT, 1, (int*)&input_tensor_dim0, (int*)&padded_size));
    CUDA_CHECK(cudnnSetTensorNdDescriptor(outputDesc, CUDNN_DATA_FLOAT, 1, (int*)&input_tensor_dim0, (int*)&padded_size));

    // Perform log_softmax
    CUDA_CHECK(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE, 
                                      inputDesc, d_padded_input, outputDesc, d_log_softmax_probs));
    CUDA_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
    CUDA_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
    CUDA_CHECK(cudnnDestroy(cudnnHandle));

    // Launch multilabel margin loss kernel
    numBlocks = (input_tensor_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    multilabel_margin_loss_kernel<<<numBlocks, threadsPerBlock>>>(d_log_softmax_probs, d_padded_input, d_loss, input_tensor_dim0);

    // Copy loss back to host
    cudaMemcpy(output, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_padded_input);
    cudaFree(d_log_softmax_probs);
    cudaFree(d_loss);
}

} // extern "C"
