
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

// CUDA kernel for ArcFace loss calculation
__global__ void arcface_loss_kernel(const float* features, const int* labels, float* output,
                                     int batch_size, int embedding_dim, float s, float m, bool easy_margin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        int label = labels[i];
        float cos_theta = 0.0f;
        for (int j = 0; j < embedding_dim; ++j) {
            cos_theta += features[i * embedding_dim + j] * features[label * embedding_dim + j];
        }
        cos_theta /= sqrtf(dot(features + i * embedding_dim, features + i * embedding_dim, embedding_dim)) *
                     sqrtf(dot(features + label * embedding_dim, features + label * embedding_dim, embedding_dim));

        // Handle numerical instability for very small cos_theta values
        cos_theta = fmaxf(fminf(cos_theta, 1.0f - 1e-7f), -1.0f + 1e-7f);

        float theta = acosf(cos_theta);
        float margin_cos_theta;
        if (easy_margin) {
            margin_cos_theta = cosf(theta + m);
            if (cos_theta - margin_cos_theta > 0.0f) {
                cos_theta = margin_cos_theta;
            }
        } else {
            cos_theta = cos_theta * (1.0f - m) + m;
        }
        output[i] = -s * cos_theta;
    }
}

__device__ float dot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

extern "C" {

void arcface_loss(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* features = va_arg(args, const float*);
    int features_dim0 = va_arg(args, int);
    int features_dim1 = va_arg(args, int);
    const int* labels = va_arg(args, const int*);
    int labels_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    // Extract additional arguments
    float s = (float) va_arg(args, double);
    float m = (float) va_arg(args, double);
    bool easy_margin = (bool) va_arg(args, int);

    va_end(args);

    int batch_size = features_dim0;
    int embedding_dim = features_dim1;

    // Allocate device memory
    float* d_features;
    int* d_labels;
    float* d_output;

    cudaMalloc(&d_features, batch_size * embedding_dim * sizeof(float));
    cudaMalloc(&d_labels, batch_size * sizeof(int));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_features, features, batch_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(128);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    arcface_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_features, d_labels, d_output, batch_size, embedding_dim, s, m, easy_margin
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_features);
    cudaFree(d_labels);
    cudaFree(d_output);
}

} // extern "C"
