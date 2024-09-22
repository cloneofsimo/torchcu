
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cutlass/cutlass.h>

#define CUDA_CHECK(condition)                                        \
  {                                                                \
    cudaError_t error = condition;                                  \
    if (error != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                        \
    }                                                              \
  }

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    int num_tensors = va_arg(args, int);
    const float** input_tensors = va_arg(args, const float**);
    int* tensor_shapes = va_arg(args, int*);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Calculate the total size of the output
    int output_size = 0;
    for (int i = 0; i < num_tensors; ++i) {
        output_size += tensor_shapes[2 * i] * tensor_shapes[2 * i + 1];
    }

    // Allocate device memory for input tensors
    float** d_input_tensors = (float**)malloc(num_tensors * sizeof(float*));
    for (int i = 0; i < num_tensors; ++i) {
        cudaMalloc(&d_input_tensors[i], tensor_shapes[2 * i] * tensor_shapes[2 * i + 1] * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_input_tensors[i], input_tensors[i], tensor_shapes[2 * i] * tensor_shapes[2 * i + 1] * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Allocate device memory for output tensor
    float* d_output;
    cudaMalloc(&d_output, output_size * sizeof(float));

    // Initialize CUDA context
    cublasHandle_t cublas_handle;
    CUDA_CHECK(cublasCreate(&cublas_handle));

    // Calculate the diagonal offsets for each tensor
    int* offsets = (int*)malloc(num_tensors * sizeof(int));
    offsets[0] = 0;
    for (int i = 1; i < num_tensors; ++i) {
        offsets[i] = offsets[i - 1] + tensor_shapes[2 * (i - 1)] * tensor_shapes[2 * (i - 1) + 1];
    }

    // Perform block diagonal matrix assembly using cublas
    for (int i = 0; i < num_tensors; ++i) {
        // Perform a copy operation using cublas
        int m = tensor_shapes[2 * i];
        int n = tensor_shapes[2 * i + 1];
        CUDA_CHECK(cublasScopy(cublas_handle, m * n, d_input_tensors[i], 1, d_output + offsets[i], 1));
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    for (int i = 0; i < num_tensors; ++i) {
        CUDA_CHECK(cudaFree(d_input_tensors[i]));
    }
    CUDA_CHECK(cudaFree(d_output));
    free(d_input_tensors);

    // Destroy CUDA context
    CUDA_CHECK(cublasDestroy(cublas_handle));
    free(offsets);
}

} // extern "C"
