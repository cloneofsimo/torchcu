
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

__global__ void isclose_int8_kernel(const int8_t* input_tensor1, const int8_t* input_tensor2, int8_t* output,
                                     int m, int n, float rtol, float atol) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float val1 = (float)input_tensor1[row * n + col];
        float val2 = (float)input_tensor2[row * n + col];
        output[row * n + col] = (abs(val1 - val2) <= atol + rtol * abs(val2)) ? 1 : 0;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const int8_t* input_tensor1 = va_arg(args, const int8_t*);
    int input_tensor1_dim0 = va_arg(args, int);
    int input_tensor1_dim1 = va_arg(args, int);

    const int8_t* input_tensor2 = va_arg(args, const int8_t*);
    int input_tensor2_dim0 = va_arg(args, int);
    int input_tensor2_dim1 = va_arg(args, int);

    // Extract rtol and atol
    float rtol = va_arg(args, double);
    float atol = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    int8_t* output = va_arg(args, int8_t*);

    va_end(args);

    int batch_size = input_tensor1_dim0;
    int input_dim = input_tensor1_dim1;

    // Allocate device memory
    int8_t *d_input1, *d_input2, *d_output;
    cudaMalloc(&d_input1, batch_size * input_dim * sizeof(int8_t));
    cudaMalloc(&d_input2, batch_size * input_dim * sizeof(int8_t));
    cudaMalloc(&d_output, batch_size * input_dim * sizeof(int8_t));

    // Copy input data to device
    cudaMemcpy(d_input1, input_tensor1, batch_size * input_dim * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input_tensor2, batch_size * input_dim * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    isclose_int8_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input1, d_input2, d_output, batch_size, input_dim, rtol, atol
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_dim * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}

}  // extern "C"
