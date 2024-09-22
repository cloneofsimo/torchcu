
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert int8 to float
__device__ __forceinline__ float int8_to_float(int8_t val) {
    return static_cast<float>(val);
}

// CUDA kernel for constant padding
__global__ void constant_pad_kernel(const int8_t* input, int8_t* output, int batch_size, int input_dim, int padding) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < input_dim + 2 * padding) {
        if (col < padding) {
            output[row * (input_dim + 2 * padding) + col] = 0; // Padding on the left
        } else if (col >= input_dim + padding) {
            output[row * (input_dim + 2 * padding) + col] = 0; // Padding on the right
        } else {
            output[row * (input_dim + 2 * padding) + col] = input[row * input_dim + col - padding];
        }
    }
}

// CUDA kernel for cosine similarity
__global__ void cosine_similarity_kernel(const int8_t* padded_input, const int8_t* weight, float* output, 
                                            int batch_size, int input_dim, int weight_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size) {
        float dot_product = 0.0f;
        float input_norm = 0.0f;
        float weight_norm = 0.0f;

        for (int i = 0; i < input_dim; ++i) {
            dot_product += int8_to_float(padded_input[row * (input_dim + 2 * padding) + i + padding]) *
                          int8_to_float(weight[i]);
            input_norm += int8_to_float(padded_input[row * (input_dim + 2 * padding) + i + padding]) *
                          int8_to_float(padded_input[row * (input_dim + 2 * padding) + i + padding]);
            weight_norm += int8_to_float(weight[i]) * int8_to_float(weight[i]);
        }

        input_norm = sqrtf(input_norm);
        weight_norm = sqrtf(weight_norm);

        if (input_norm * weight_norm > 0.0f) {
            output[row] = dot_product / (input_norm * weight_norm);
        } else {
            output[row] = 0.0f; // Handle the case where norms are zero
        }
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

    // Extract weight tensor
    const int8_t* weight = va_arg(args, const int8_t*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract padding value
    int padding = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int weight_dim = weight_dim0; // Assuming weight is a single vector

    // Allocate device memory
    int8_t *d_input, *d_weight, *d_padded_input;
    float *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(int8_t));
    cudaMalloc(&d_weight, weight_dim * sizeof(int8_t));
    cudaMalloc(&d_padded_input, batch_size * (input_dim + 2 * padding) * sizeof(int8_t));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_dim * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch constant padding kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((input_dim + 2 * padding + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    constant_pad_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_padded_input, batch_size, input_dim, padding);

    // Launch cosine similarity kernel
    dim3 threadsPerBlock_cos(128);
    dim3 numBlocks_cos((batch_size + threadsPerBlock_cos.x - 1) / threadsPerBlock_cos.x);
    cosine_similarity_kernel<<<numBlocks_cos, threadsPerBlock_cos>>>(
        d_padded_input, d_weight, d_output, batch_size, input_dim, weight_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_padded_input);
    cudaFree(d_output);
}

}  // extern "C"
