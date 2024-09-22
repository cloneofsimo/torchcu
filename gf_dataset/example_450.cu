
#include <cuda_runtime.h>
#include <cudnn.h>

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    const float* vec = va_arg(args, const float*);
    int vec_dim = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Check if input dimensions are compatible
    if (input_tensor_dim1 != weight_dim1 || weight_dim0 != vec_dim) {
        // Handle error: incompatible dimensions
        return;
    }

    // Cudnn setup
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);

    // Allocate device memory
    float *d_input, *d_weight, *d_vec, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
    cudaMalloc(&d_weight, weight_dim0 * weight_dim1 * sizeof(float));
    cudaMalloc(&d_vec, vec_dim * sizeof(float));
    cudaMalloc(&d_output, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim0 * weight_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vec, vec_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Create cudnn tensor descriptors
    cudnnTensorDescriptor_t input_tensor_desc, weight_desc, vec_desc, output_desc;
    cudnnCreateTensorDescriptor(&input_tensor_desc);
    cudnnCreateTensorDescriptor(&weight_desc);
    cudnnCreateTensorDescriptor(&vec_desc);
    cudnnCreateTensorDescriptor(&output_desc);

    // Set tensor descriptors
    cudnnSetTensorNdDescriptor(input_tensor_desc, CUDNN_DATA_FLOAT, 2,
                                 (const int[]){input_tensor_dim0, input_tensor_dim1});
    cudnnSetTensorNdDescriptor(weight_desc, CUDNN_DATA_FLOAT, 2,
                                 (const int[]){weight_dim0, weight_dim1});
    cudnnSetTensorNdDescriptor(vec_desc, CUDNN_DATA_FLOAT, 1,
                                 (const int[]){vec_dim});
    cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, 2,
                                 (const int[]){input_tensor_dim0, input_tensor_dim1});

    // Perform addmv operation
    cudnnAddTensor(cudnn_handle, CUDNN_ADD_SAME_ALPHA_BETA,
                      &one, input_tensor_desc, d_input,
                      &one, weight_desc, d_weight, 
                      &one, vec_desc, d_vec, 
                      &one, output_desc, d_output);

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free resources
    cudnnDestroyTensorDescriptor(input_tensor_desc);
    cudnnDestroyTensorDescriptor(weight_desc);
    cudnnDestroyTensorDescriptor(vec_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_vec);
    cudaFree(d_output);
    cudnnDestroy(cudnn_handle);
}

}  // extern "C"
