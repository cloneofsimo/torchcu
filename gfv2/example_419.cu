
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

#define BLOCK_SIZE 16

// Helper function for ReLU activation
__device__ inline float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

// CUDA kernel for matrix multiplication and ReLU
__global__ void matmul_relu_kernel(const float* input_tensor, const float* weight1, const float* weight2, float* output, 
                                    int batch_size, int input_dim, int hidden_dim, int output_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < output_dim) {
        float sum1 = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            sum1 += input_tensor[row * input_dim + i] * weight1[col * input_dim + i];
        }
        float sum2 = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) {
            sum2 += relu(sum1) * weight2[col * hidden_dim + i];
        }
        output[row * output_dim + col] = sum2;
    }
}

// CUDA kernel for bmm_out (batch matrix multiplication)
__global__ void bmm_out_kernel(const float* input_tensor, const float* block_diagonal, float* output, 
                                int batch_size, int input_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < input_dim) {
        for (int i = 0; i < input_dim; ++i) {
            output[row * input_dim + col] += input_tensor[row * input_dim + i] * block_diagonal[col * input_dim + i];
        }
    }
}

// CUDA kernel for MSE loss calculation
__global__ void mse_loss_kernel(const float* output, const float* target, float* loss, 
                                int batch_size, int output_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < output_dim) {
        float diff = output[row * output_dim + col] - target[row * output_dim + col];
        atomicAdd(loss, diff * diff);
    }
}

extern "C" {

void my_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract hyperparameters
    float learning_rate = va_arg(args, double);
    float weight_decay = va_arg(args, double);
    bool use_fp16 = va_arg(args, int);

    // Extract target tensor (optional)
    const float* target_tensor = va_arg(args, const float*);
    int target_tensor_dim0 = va_arg(args, int);
    int target_tensor_dim1 = va_arg(args, int);

    // Extract output tensors (assuming they are preallocated)
    float* output = va_arg(args, float*);
    float* loss = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int hidden_dim = 64;
    int output_dim = 32;

    // Allocate device memory
    float *d_input, *d_weight1, *d_weight2, *d_output, *d_target, *d_block_diagonal;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight1, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_weight2, output_dim * hidden_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));
    cudaMalloc(&d_block_diagonal, batch_size * input_dim * input_dim * sizeof(float));

    if (target_tensor != nullptr) {
        cudaMalloc(&d_target, batch_size * output_dim * sizeof(float));
    }

    // Initialize weights on device
    float *h_weight1 = new float[output_dim * input_dim];
    float *h_weight2 = new float[output_dim * hidden_dim];
    for (int i = 0; i < output_dim * input_dim; ++i) {
        h_weight1[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < output_dim * hidden_dim; ++i) {
        h_weight2[i] = (float)rand() / RAND_MAX;
    }
    cudaMemcpy(d_weight1, h_weight1, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight2, h_weight2, output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_weight1;
    delete[] h_weight2;

    // Create block diagonal matrix on device
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            d_block_diagonal[i * input_dim * input_dim + j * input_dim + j] = 1.0f;
        }
    }

    // Copy input and target data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    if (target_tensor != nullptr) {
        cudaMemcpy(d_target, target_tensor, batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Perform bmm_out on device
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((input_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    bmm_out_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_block_diagonal, d_output, batch_size, input_dim);

    // Perform forward pass on device
    numBlocks = ((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_relu_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_weight1, d_weight2, d_output, 
                                                batch_size, input_dim, hidden_dim, output_dim);

    // Calculate loss on device if target tensor is provided
    if (target_tensor != nullptr) {
        numBlocks = ((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        mse_loss_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_target, loss, batch_size, output_dim);
    }

    // Copy output and loss data back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);
    if (target_tensor != nullptr) {
        cudaMemcpy(loss, loss, sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight1);
    cudaFree(d_weight2);
    cudaFree(d_output);
    cudaFree(d_block_diagonal);
    if (target_tensor != nullptr) {
        cudaFree(d_target);
    }
}

} // extern "C"

