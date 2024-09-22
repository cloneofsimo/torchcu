
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void sobel_gradient_int8_kernel(const int8_t *input, int8_t *output, 
                                        int batch, int channels, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = (y * width + x) + (batch * height * width);

    // Check if within bounds
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        // Sobel x kernel
        int8_t grad_x = input[index - width - 1] + 2 * input[index - width] + input[index - width + 1]
                       - input[index + width - 1] - 2 * input[index + width] - input[index + width + 1];
        // Sobel y kernel
        int8_t grad_y = input[index - width - 1] + 2 * input[index - 1] + input[index + width - 1]
                       - input[index - width + 1] - 2 * input[index + 1] - input[index + width + 1];
        // Store gradient in output tensor
        output[index] = grad_x;
        output[index + batch * height * width * channels] = grad_y;
    }
}

extern "C" {
    void sobel_gradient_int8_function(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract input tensor
        const float *input_tensor = va_arg(args, const float *);
        int input_tensor_dim0 = va_arg(args, int);
        int input_tensor_dim1 = va_arg(args, int);
        int input_tensor_dim2 = va_arg(args, int);
        int input_tensor_dim3 = va_arg(args, int);

        // Extract output tensor (assuming it's preallocated)
        int8_t *output = va_arg(args, int8_t *);

        va_end(args);

        // Allocate device memory
        int8_t *d_input, *d_output;
        cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(int8_t));
        cudaMalloc(&d_output, input_tensor_dim0 * input_tensor_dim1 * 2 * input_tensor_dim2 * input_tensor_dim3 * sizeof(int8_t));

        // Copy input data to device
        cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((input_tensor_dim3 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (input_tensor_dim2 + threadsPerBlock.y - 1) / threadsPerBlock.y);
        sobel_gradient_int8_kernel<<<numBlocks, threadsPerBlock>>>(
            d_input, d_output, input_tensor_dim0, input_tensor_dim1, input_tensor_dim2, input_tensor_dim3
        );

        // Copy result back to host
        cudaMemcpy(output, d_output, input_tensor_dim0 * input_tensor_dim1 * 2 * input_tensor_dim2 * input_tensor_dim3 * sizeof(int8_t), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
    }
}
