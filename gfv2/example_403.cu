
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 
#include <stdio.h>

// CUDA kernel for simple data processing
__global__ void process_data_kernel(const float* input_tensor, float* output, float sum_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input_tensor[idx] * sum_data;
    }
}

// Helper function to load data from a file
float load_data_sum(const char* filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return 0.0f;
    }

    float sum = 0.0f;
    float data_value;
    while (fread(&data_value, sizeof(float), 1, fp) == 1) {
        sum += data_value;
    }

    fclose(fp);
    return sum;
}

extern "C" {

void process_data(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int); 

    // Extract filename (dummy type for now)
    const char* filename = va_arg(args, const char*);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int size = input_tensor_dim0;

    // Load data from file and calculate sum
    float sum_data = load_data_sum(filename);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    process_data_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, sum_data, size);

    // Copy result back to host (convert to fp16)
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        output[i] = __float2half_rn(output[i]); // Convert to fp16
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
