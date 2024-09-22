
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for element-wise sum with mode
__global__ void elementwise_sum_kernel(const float* input_tensors[], float* output,
                                        int num_tensors, int batch_size, int num_elements, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * num_elements) {
        float sum = input_tensors[0][i];
        for (int t = 1; t < num_tensors; ++t) {
            float val = input_tensors[t][i];
            if (mode == 0) { // le
                sum = (val <= sum) ? sum : val;
            } else if (mode == 1) { // ge
                sum = (val >= sum) ? sum : val;
            }
            sum += val;
        }
        output[i] = sum;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    int num_tensors = va_arg(args, int);
    const float** input_tensors = va_arg(args, const float**);
    int batch_size = va_arg(args, int);
    int num_elements = va_arg(args, int);

    // Extract mode
    int mode = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_output;
    cudaMalloc(&d_output, batch_size * num_elements * sizeof(float));

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size * num_elements + threadsPerBlock.x - 1) / threadsPerBlock.x);
    elementwise_sum_kernel<<<numBlocks, threadsPerBlock>>>(input_tensors, d_output,
                                                                num_tensors, batch_size, num_elements, mode);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_output);
}

} // extern "C"
