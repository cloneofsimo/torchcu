
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fold_kernel(const float* input_tensor, const float* initial_value, float* output,
                            int batch_size, int height, int width, int axis) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < width) {
        int input_idx = row * width * height + col * height + axis;
        output[row * width + col] = initial_value[0];
        for (int i = 0; i < height; ++i) {
            output[row * width + col] += input_tensor[input_idx];
            input_idx += width;
        }
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract arguments
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int axis = va_arg(args, int);
    const float* initial_value = va_arg(args, const float*);
    int initial_value_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int height = input_tensor_dim1;
    int width = input_tensor_dim2;

    // Allocate device memory
    float *d_input, *d_initial_value, *d_output;
    cudaMalloc(&d_input, batch_size * height * width * sizeof(float));
    cudaMalloc(&d_initial_value, initial_value_dim0 * sizeof(float));
    cudaMalloc(&d_output, batch_size * width * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, batch_size * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_initial_value, initial_value, initial_value_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    fold_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_initial_value, d_output, batch_size, height, width, axis
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_initial_value);
    cudaFree(d_output);
}

}  // extern "C"
