
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// CUDA kernel for cross-fade operation
__global__ void cross_fade_kernel(const float* input1, const float* input2, float* output, float alpha, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float val1 = input1[row * n + col];
        float val2 = input2[row * n + col];
        output[row * n + col] = (1 - alpha) * val1 + alpha * val2; 
    }
}

// CUDA kernel for thresholding and addcdiv operation
__global__ void threshold_addcdiv_kernel(int8_t* output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int8_t val = output[row * n + col];
        if (val != 0) {
            output[row * n + col] = val + 1;
        }
    }
}

extern "C" {

void cross_fade_int8_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input1 = va_arg(args, const float*);
    int input1_dim0 = va_arg(args, int);
    int input1_dim1 = va_arg(args, int);
    const float* input2 = va_arg(args, const float*);
    int input2_dim0 = va_arg(args, int);
    int input2_dim1 = va_arg(args, int);
    const float* alpha_ptr = va_arg(args, const float*); // alpha is a single float
    float alpha = *alpha_ptr; // dereference the pointer to get the value

    // Extract output tensor (assuming it's preallocated)
    int8_t* output = va_arg(args, int8_t*);

    va_end(args);

    int batch_size = input1_dim0;
    int input_dim = input1_dim1;

    // Allocate device memory
    float *d_input1, *d_input2;
    int8_t *d_output;
    cudaMalloc(&d_input1, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_input2, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim * sizeof(int8_t));

    // Copy input data to device
    cudaMemcpy(d_input1, input1, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch cross-fade kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cross_fade_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input1, d_input2, d_output, alpha, batch_size, input_dim
    );

    // Launch thresholding and addcdiv kernel
    threshold_addcdiv_kernel<<<numBlocks, threadsPerBlock>>>(
        d_output, batch_size, input_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_dim * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}

}  // extern "C"
