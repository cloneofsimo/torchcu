
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void scharr_gradient_kernel(const float* input, float* output, 
                                      int batch_size, int channels, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) + (blockIdx.z * height * width);

    if (x < width && y < height && idx < batch_size * channels * height * width) {
        // Scharr filter coefficients
        int scharr_x[3][3] = {{-3, 0, 3}, {-10, 0, 10}, {-3, 0, 3}};
        int scharr_y[3][3] = {{3, 10, 3}, {0, 0, 0}, {-3, -10, -3}};

        float sum_x = 0.0f;
        float sum_y = 0.0f;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int neighbor_x = x + j;
                int neighbor_y = y + i;
                if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                    int neighbor_idx = (neighbor_y * width + neighbor_x) + (blockIdx.z * height * width);
                    sum_x += input[neighbor_idx] * scharr_x[i + 1][j + 1];
                    sum_y += input[neighbor_idx] * scharr_y[i + 1][j + 1];
                }
            }
        }
        output[idx] = sqrtf(sum_x * sum_x + sum_y * sum_y); // Calculate gradient magnitude
    }
}

__global__ void nll_loss_kernel(const float* input, const int* target, float* loss, 
                               int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        int target_idx = target[idx];
        int input_idx = target_idx * channels * height * width + (blockIdx.y * height * width + (blockIdx.z * width + threadIdx.y));
        loss[idx] = -logf(input[input_idx]); // Assuming softmax is already applied
    }
}

__global__ void rmse_kernel(const float* input, const float* target, float* rmse, 
                           int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        int offset = (idx * channels * height * width);
        float sum_sq_error = 0.0f;
        for (int i = 0; i < channels * height * width; ++i) {
            sum_sq_error += (input[offset + i] - target[offset + i]) * (input[offset + i] - target[offset + i]);
        }
        rmse[idx] = sqrtf(sum_sq_error / (channels * height * width));
    }
}

extern "C" {
    void compute_nll_loss_with_rmse(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract input tensor
        const float* input = va_arg(args, const float*);
        int input_batch = va_arg(args, int);
        int input_channels = va_arg(args, int);
        int input_height = va_arg(args, int);
        int input_width = va_arg(args, int);

        // Extract target tensor
        const int* target = va_arg(args, const int*);
        int target_size = va_arg(args, int);

        // Extract output tensors (assuming they are pre-allocated)
        float* nll_loss = va_arg(args, float*);
        float* rmse = va_arg(args, float*);

        va_end(args);

        // Allocate device memory for input and target
        float* d_input;
        int* d_target;
        cudaMalloc(&d_input, input_batch * input_channels * input_height * input_width * sizeof(float));
        cudaMalloc(&d_target, target_size * sizeof(int));

        // Copy input and target tensors to device
        cudaMemcpy(d_input, input, input_batch * input_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_target, target, target_size * sizeof(int), cudaMemcpyHostToDevice);

        // Apply Scharr gradient
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((input_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (input_height + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                       input_batch);
        scharr_gradient_kernel<<<numBlocks, threadsPerBlock>>>(
            d_input, d_input, input_batch, input_channels, input_height, input_width
        );

        // Calculate NLL loss
        dim3 nll_blocks((target_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
        nll_loss_kernel<<<nll_blocks, threadsPerBlock>>>(
            d_input, d_target, nll_loss, input_batch, input_channels, input_height, input_width
        );

        // Calculate RMSE
        dim3 rmse_blocks((input_batch + threadsPerBlock.x - 1) / threadsPerBlock.x);
        rmse_kernel<<<rmse_blocks, threadsPerBlock>>>(
            d_input, d_input, rmse, input_batch, input_channels, input_height, input_width
        );

        // Copy results back to host
        cudaMemcpy(nll_loss, nll_loss, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(rmse, rmse, sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_target);
    }
}
