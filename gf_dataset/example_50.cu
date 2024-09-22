
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for sorting and computing median
__global__ void sort_median_kernel(const float* input, float* output, int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Compute the index within the dimension to sort
        int dim_idx = idx % dim;

        // Compute the starting index of the current slice along the specified dimension
        int slice_start = idx - dim_idx;

        // Create a temporary array to store the slice
        float temp[dim];

        // Copy the slice from input to temp
        for (int i = 0; i < dim; ++i) {
            temp[i] = input[slice_start + i];
        }

        // Sort the temp array
        for (int i = 0; i < dim - 1; ++i) {
            for (int j = i + 1; j < dim; ++j) {
                if (temp[i] > temp[j]) {
                    float tmp = temp[i];
                    temp[i] = temp[j];
                    temp[j] = tmp;
                }
            }
        }

        // Copy the sorted slice back to input
        for (int i = 0; i < dim; ++i) {
            input[slice_start + i] = temp[i];
        }

        // Calculate the median index
        int median_idx = dim / 2;

        // Copy the median value to output
        output[idx] = temp[median_idx];
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

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    // Extract dimension
    int dim = va_arg(args, int);

    va_end(args);

    int N = input_tensor_dim0 * input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    sort_median_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N, dim);

    // Copy result back to host
    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
