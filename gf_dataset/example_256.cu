
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>
#include <cudnn.h>

#define CHECK_CUDNN(status)                                      \
  do {                                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                       \
      const char* msg;                                          \
      cudnnGetErrorString(status, &msg);                         \
      fprintf(stderr, "CUDNN error: %s\n", msg);               \
      exit(EXIT_FAILURE);                                       \
    }                                                          \
  } while (0)

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract output tensors (assuming they are preallocated)
    float* max_values = va_arg(args, float*);
    float* energy = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_max_values, *d_energy;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_max_values, batch_size * sizeof(float));
    cudaMalloc(&d_energy, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize CUDNN
    cudnnHandle_t cudnnHandle;
    CHECK_CUDNN(cudnnCreate(&cudnnHandle));

    // Create CUDNN tensor descriptors
    cudnnTensorDescriptor_t inputDesc, maxValuesDesc, energyDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&maxValuesDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&energyDesc));

    // Set tensor descriptors
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_DATA_FLOAT, 1, batch_size, 1, input_dim));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(maxValuesDesc, CUDNN_DATA_FLOAT, 1, batch_size, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(energyDesc, CUDNN_DATA_FLOAT, 1, batch_size, 1, 1));

    // Perform the maximum operation using CUDNN
    cudnnReduceTensorDescriptor_t reduceDesc;
    CHECK_CUDNN(cudnnCreateReduceTensorDescriptor(&reduceDesc));
    CHECK_CUDNN(cudnnSetReduceTensorDescriptor(reduceDesc, CUDNN_REDUCE_MAX, CUDNN_DATA_FLOAT, CUDNN_REDUCE_NO_INDICES));

    CHECK_CUDNN(cudnnReduceTensor(cudnnHandle, reduceDesc, 1.0f, inputDesc, d_input, 1.0f, maxValuesDesc, d_max_values));

    // Perform the energy computation
    CHECK_CUDNN(cudnnSetReduceTensorDescriptor(reduceDesc, CUDNN_REDUCE_SUM, CUDNN_DATA_FLOAT, CUDNN_REDUCE_NO_INDICES));
    CHECK_CUDNN(cudnnReduceTensor(cudnnHandle, reduceDesc, 1.0f, inputDesc, d_input, 1.0f, energyDesc, d_energy));

    // Copy results back to host
    cudaMemcpy(max_values, d_max_values, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(energy, d_energy, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free CUDNN resources
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(maxValuesDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(energyDesc));
    CHECK_CUDNN(cudnnDestroyReduceTensorDescriptor(reduceDesc));
    CHECK_CUDNN(cudnnDestroy(cudnnHandle));

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_max_values);
    cudaFree(d_energy);
}

}  // extern "C"

