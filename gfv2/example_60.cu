
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for adaptive max pooling 1D
__global__ void adaptive_max_pool1d_kernel(const float* input, float* output, int batch_size, int channels, int time_steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * channels) {
        int batch = idx / channels;
        int channel = idx % channels;

        float max_value = input[batch * channels * time_steps + channel];
        for (int t = 1; t < time_steps; ++t) {
            float current_value = input[batch * channels * time_steps + channel + t * channels];
            max_value = fmaxf(max_value, current_value);
        }
        output[idx] = max_value;
    }
}

// CUDA kernel for contrastive loss gradient computation
__global__ void contrastive_loss_gradient_kernel(const float* pooled_output, const float* labels, float* gradients, 
                                                 int batch_size, int channels, float temperature) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * channels) {
        int batch = idx / channels;
        int channel = idx % channels;
        float output_value = pooled_output[idx];
        float label_value = labels[batch];

        float similarity = output_value * output_value;
        float loss_gradient = (similarity - label_value) / (batch_size * temperature);
        gradients[idx] = loss_gradient;
    }
}

extern "C" {

void contrastive_pooling(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);
    int input_dim2 = va_arg(args, int);

    // Extract labels tensor
    const float* labels = va_arg(args, const float*);
    int labels_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);
    float* gradients = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float* d_input, *d_output, *d_labels, *d_gradients;
    cudaMalloc(&d_input, input_dim0 * input_dim1 * input_dim2 * sizeof(float));
    cudaMalloc(&d_output, input_dim0 * input_dim1 * sizeof(float));
    cudaMalloc(&d_labels, labels_dim0 * sizeof(float));
    cudaMalloc(&d_gradients, input_dim0 * input_dim1 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, input_dim0 * input_dim1 * input_dim2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, labels_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch adaptive max pooling kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((input_dim0 * input_dim1 + threadsPerBlock.x - 1) / threadsPerBlock.x);
    adaptive_max_pool1d_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, input_dim0, input_dim1, input_dim2
    );

    // Launch contrastive loss gradient computation kernel
    float temperature = 0.1f;
    contrastive_loss_gradient_kernel<<<numBlocks, threadsPerBlock>>>(
        d_output, d_labels, d_gradients, input_dim0, input_dim1, temperature
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, input_dim0 * input_dim1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gradients, d_gradients, input_dim0 * input_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_labels);
    cudaFree(d_gradients);
}

}
