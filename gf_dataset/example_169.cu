
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h> 
#include <stdarg.h>

// CUDA kernel for matrix multiplication and Mish activation using int8
__global__ void matmul_mish_kernel_int8(const int8_t* input_tensor, const int8_t* weight, int8_t* output, 
                                        int m, int n, int k, float scale_in, float scale_out, float scale_weight) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += (float)input_tensor[row * k + i] * (float)weight[col * k + i] * scale_in * scale_weight;
        }

        // Mish activation
        float x = sum * scale_out;
        output[row * n + col] = (int8_t)(tanh(logf(1.0f + expf(x))) * x / scale_out);
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const int8_t* input_tensor = va_arg(args, const int8_t*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    float scale_in = va_arg(args, float);

    // Extract weight tensor
    const int8_t* weight = va_arg(args, const int8_t*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    float scale_weight = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    int8_t* output = va_arg(args, int8_t*);
    float scale_out = va_arg(args, float); 

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;

    // Allocate device memory
    int8_t *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(int8_t));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(int8_t));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(int8_t));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_mish_kernel_int8<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, batch_size, output_dim, input_dim, scale_in, scale_out, scale_weight
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
