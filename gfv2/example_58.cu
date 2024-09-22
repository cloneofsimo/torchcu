
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for cross-entropy with bucketized weights
__global__ void cross_entropy_kernel(const float* input, const int* target, const float* weights, 
                                     const float* bucket_boundaries, float* loss, int batch_size, int num_classes, int num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int target_class = target[idx];
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            sum_exp += expf(input[idx * num_classes + c]);
        }
        float log_sum_exp = logf(sum_exp);
        float input_val = input[idx * num_classes + target_class];
        float loss_val = log_sum_exp - input_val;

        // Bucketize weights on the device
        int bucket_idx = 0;
        while (bucket_idx < num_buckets - 1 && weights[target_class] > bucket_boundaries[bucket_idx]) {
            bucket_idx++;
        }
        loss[idx] = loss_val * bucket_idx; 
    }
}

extern "C" {

void cross_entropy_with_bucketized_weights(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int num_classes = va_arg(args, int);

    const int* target = va_arg(args, const int*);

    const float* weights = va_arg(args, const float*);

    int num_buckets = va_arg(args, int);

    float* output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_weights, *d_loss, *d_bucket_boundaries;
    int *d_target;
    cudaMalloc(&d_input, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_weights, num_classes * sizeof(float));
    cudaMalloc(&d_loss, batch_size * sizeof(float));
    cudaMalloc(&d_target, batch_size * sizeof(int));
    cudaMalloc(&d_bucket_boundaries, (num_buckets - 1) * sizeof(float)); 

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, num_classes * sizeof(float), cudaMemcpyHostToDevice);

    // Generate bucket boundaries on the device
    float* bucket_boundaries = new float[num_buckets - 1]; 
    for (int i = 0; i < num_buckets - 1; i++) {
        bucket_boundaries[i] = weights[0] + (weights[num_classes - 1] - weights[0]) * (i + 1.0f) / num_buckets;
    }
    cudaMemcpy(d_bucket_boundaries, bucket_boundaries, (num_buckets - 1) * sizeof(float), cudaMemcpyHostToDevice);
    delete[] bucket_boundaries;

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    cross_entropy_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_target, d_weights, d_bucket_boundaries, d_loss, batch_size, num_classes, num_buckets
    );

    // Copy result back to host
    cudaMemcpy(output, d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_loss);
    cudaFree(d_target);
    cudaFree(d_bucket_boundaries);
}

}  // extern "C"
