
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function for sigmoid focal loss
__device__ __forceinline__ float sigmoid_focal_loss_kernel(float p, float y, float alpha, float gamma) {
    float pt = y * p + (1.0f - y) * (1.0f - p);
    float loss = alpha * powf(1.0f - pt, gamma) * (-logf(pt));
    return loss;
}

// Helper function for cross-entropy loss
__device__ __forceinline__ float cross_entropy_kernel(float p, int y, const float* weights) {
    if (y == 0) {
        return -logf(1.0f - p) * weights[0];
    } else {
        return -logf(p) * weights[1];
    }
}

// CUDA kernel for combined loss calculations
__global__ void combined_loss_kernel(const float* input_tensor, const int* target_tensor, const float* class_weights, float* loss, 
                                       int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * num_classes) {
        int batch_index = idx / num_classes;
        int class_index = idx % num_classes;
        float p = input_tensor[idx];
        int y = target_tensor[idx];
        
        float mse_loss = powf(p - y, 2.0f);
        float sigmoid_focal_loss = sigmoid_focal_loss_kernel(p, y, 0.25f, 2.0f); // Example alpha and gamma
        float cross_entropy_loss = cross_entropy_kernel(p, y, class_weights);

        loss[batch_index] += mse_loss + sigmoid_focal_loss + cross_entropy_loss; 
    }
}

extern "C" {

void multi_loss_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    
    const int* target_tensor = va_arg(args, const int*);
    int target_tensor_dim0 = va_arg(args, int);
    int target_tensor_dim1 = va_arg(args, int);

    // Extract class weights
    const float* class_weights = va_arg(args, const float*);
    int class_weights_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* loss = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int num_classes = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_class_weights;
    int *d_target;
    cudaMalloc(&d_input, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_target, batch_size * num_classes * sizeof(int));
    cudaMalloc(&d_class_weights, num_classes * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, batch_size * num_classes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_class_weights, class_weights, num_classes * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize output array on device
    cudaMemset(loss, 0, batch_size * sizeof(float));

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size * num_classes + threadsPerBlock.x - 1) / threadsPerBlock.x);
    combined_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_target, d_class_weights, loss, batch_size, num_classes
    );

    // Copy result back to host
    cudaMemcpy(loss, loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_class_weights);
}

} // extern "C"
