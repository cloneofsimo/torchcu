
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void softmin_kernel(const float* input, float* output, int batch_size, int input_dim, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < input_dim) {
        float max_val = input[row * input_dim + col];
        float sum_exp = 0.0f;

        for (int i = 0; i < input_dim; ++i) {
            if (dim == 0 && i != col) {
                max_val = fmaxf(max_val, input[i * input_dim + row]);
            } else if (dim == 1 && i != row) {
                max_val = fmaxf(max_val, input[row * input_dim + i]);
            }
        }

        for (int i = 0; i < input_dim; ++i) {
            if (dim == 0 && i != col) {
                sum_exp += expf(-input[i * input_dim + row] + max_val);
            } else if (dim == 1 && i != row) {
                sum_exp += expf(-input[row * input_dim + i] + max_val);
            }
        }

        if (dim == 0) {
            output[row * input_dim + col] = expf(-input[row * input_dim + col] + max_val) / sum_exp;
        } else if (dim == 1) {
            output[row * input_dim + col] = expf(-input[row * input_dim + col] + max_val) / sum_exp;
        }
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

    // Extract dimension
    int dim = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    softmin_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, batch_size, input_dim, dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
