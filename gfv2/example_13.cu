
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

extern "C" {

void center_loss_with_adaptive_pooling(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* features = va_arg(args, const float*);
    int features_dim0 = va_arg(args, int);
    int features_dim1 = va_arg(args, int);

    const int* labels = va_arg(args, const int*);
    int labels_dim0 = va_arg(args, int);

    int num_classes = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_features, *d_centers, *d_output;
    int *d_labels;
    cudaMalloc(&d_features, features_dim0 * features_dim1 * sizeof(float));
    cudaMalloc(&d_centers, num_classes * features_dim1 * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMalloc(&d_labels, labels_dim0 * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_features, features, features_dim0 * features_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, labels_dim0 * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize Cudnn
    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);

    // Create Cudnn descriptors for pooling
    cudnnPoolingDescriptor_t poolingDesc;
    cudnnCreatePoolingDescriptor(&poolingDesc);
    cudnnSetPoolingDescriptor(poolingDesc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, 
                              CUDNN_PROPAGATE_NAN, 1, 1, 1);

    // Create Cudnn descriptors for tensors
    cudnnTensorDescriptor_t featuresDesc, centersDesc;
    cudnnCreateTensorDescriptor(&featuresDesc);
    cudnnCreateTensorDescriptor(&centersDesc);
    cudnnSetTensorDescriptor(featuresDesc, CUDNN_DATA_TYPE_FLOAT, 1, features_dim1, 1, 1, 1, 1);
    cudnnSetTensorDescriptor(centersDesc, CUDNN_DATA_TYPE_FLOAT, 1, features_dim1, 1, 1, 1, 1);

    // Perform adaptive average pooling
    cudnnPoolingForward(cudnnHandle, poolingDesc, featuresDesc, d_features, featuresDesc, d_features);

    // Allocate and initialize centers
    float* centers = new float[num_classes * features_dim1];
    for (int i = 0; i < num_classes * features_dim1; ++i) {
        centers[i] = (float)rand() / RAND_MAX; // Initialize randomly
    }
    cudaMemcpy(d_centers, centers, num_classes * features_dim1 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to calculate center loss
    const float alpha = 0.5f;
    cudaLaunchKernel((const void*)center_loss_kernel, // Function pointer
                       (features_dim0 + 255) / 256, // Blocks per grid (x)
                       1, // Blocks per grid (y)
                       1, // Blocks per grid (z)
                       256, // Threads per block (x)
                       1, // Threads per block (y)
                       1, // Threads per block (z)
                       0, // Shared memory size (bytes)
                       NULL, // Stream
                       d_features, // Input feature tensor
                       d_centers, // Class centers tensor
                       d_labels, // Labels tensor
                       features_dim1, // Feature dimension
                       num_classes, // Number of classes
                       d_output, // Output loss value
                       alpha // Weighting factor
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_features);
    cudaFree(d_centers);
    cudaFree(d_labels);
    cudaFree(d_output);

    // Free Cudnn descriptors
    cudnnDestroyPoolingDescriptor(poolingDesc);
    cudnnDestroyTensorDescriptor(featuresDesc);
    cudnnDestroyTensorDescriptor(centersDesc);

    // Destroy Cudnn handle
    cudnnDestroy(cudnnHandle);

    delete[] centers;
}

// CUDA kernel to calculate the center loss
__global__ void center_loss_kernel(const float* features, const float* centers, const int* labels, 
                                    int feature_dim, int num_classes, float* loss, float alpha) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < blockDim.x * gridDim.x) {
        int label = labels[index];
        float* center = centers + label * feature_dim;
        float sum = 0.0f;
        for (int i = 0; i < feature_dim; ++i) {
            float diff = features[index * feature_dim + i] - center[i];
            sum += diff * diff;
        }
        atomicAdd(loss, alpha * sum / (float)(blockDim.x * gridDim.x));
    }
}

}  // extern "C"
