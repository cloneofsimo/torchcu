
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// Helper functions for FP16 operations
__device__ __forceinline__ half rrelu_activation(half x, half lower, half upper) {
    return x > 0 ? x : (x * lower + (upper - lower) * x * (x < 0));
}

__device__ __forceinline__ half smooth_l1_loss(half x) {
    return x < 1.0f ? 0.5f * x * x : abs(x) - 0.5f;
}

__device__ __forceinline__ half poisson_loss(half x, half y) {
    return x - y + y * log(y / x);
}

__global__ void multi_loss_kernel(const float* input_tensor, const float* target_tensor, const float* center_weights,
                                  float* smooth_l1_loss, float* rrelu_loss, float* poisson_loss, float* center_loss,
                                  float alpha, float beta, float gamma, int batch_size, int num_centers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        // Convert inputs to FP16
        half input = __float2half_rn(input_tensor[idx]);
        half target = __float2half_rn(target_tensor[idx]);

        // Smooth L1 Loss
        smooth_l1_loss[idx] = __half2float(smooth_l1_loss(input - target));

        // RReLU Loss
        half rrelu = rrelu_activation(input, __float2half_rn(0.1f), __float2half_rn(0.3f));
        rrelu_loss[idx] = __half2float(abs(rrelu - target));

        // Poisson Loss
        poisson_loss[idx] = __half2float(poisson_loss(input, target));

        // Center Loss
        float center_sum = 0.0f;
        for (int j = 0; j < num_centers; ++j) {
            half center = __float2half_rn(center_weights[idx * num_centers + j]);
            center_sum += __half2float((input - center) * (input - center));
        }
        center_loss[idx] = center_sum;

        // Combine and store losses
        smooth_l1_loss[idx] *= alpha;
        rrelu_loss[idx] *= beta;
        poisson_loss[idx] *= gamma;
    }
}

extern "C" {

void multi_loss_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input arguments
    const float* input_tensor = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);

    const float* target_tensor = va_arg(args, const float*);
    int target_dim0 = va_arg(args, int);
    int target_dim1 = va_arg(args, int);

    const float* center_weights = va_arg(args, const float*);
    int center_dim0 = va_arg(args, int);
    int center_dim1 = va_arg(args, int);

    float alpha = (float)va_arg(args, double);
    float beta = (float)va_arg(args, double);
    float gamma = (float)va_arg(args, double);

    // Extract output tensors
    float* smooth_l1_loss = va_arg(args, float*);
    float* rrelu_loss = va_arg(args, float*);
    float* poisson_loss = va_arg(args, float*);
    float* center_loss = va_arg(args, float*);
    float* total_loss = va_arg(args, float*);

    va_end(args);

    int batch_size = input_dim0;
    int num_centers = center_dim1;

    // Allocate device memory
    float *d_input, *d_target, *d_center, *d_smooth_l1, *d_rrelu, *d_poisson, *d_center_loss;
    cudaMalloc(&d_input, batch_size * sizeof(float));
    cudaMalloc(&d_target, batch_size * sizeof(float));
    cudaMalloc(&d_center, batch_size * num_centers * sizeof(float));
    cudaMalloc(&d_smooth_l1, batch_size * sizeof(float));
    cudaMalloc(&d_rrelu, batch_size * sizeof(float));
    cudaMalloc(&d_poisson, batch_size * sizeof(float));
    cudaMalloc(&d_center_loss, batch_size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_center, center_weights, batch_size * num_centers * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    multi_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_target, d_center, d_smooth_l1, d_rrelu, d_poisson, d_center_loss,
        alpha, beta, gamma, batch_size, num_centers
    );

    // Copy results back to host
    cudaMemcpy(smooth_l1_loss, d_smooth_l1, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rrelu_loss, d_rrelu, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(poisson_loss, d_poisson, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_loss, d_center_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate total loss and store
    float total_loss_sum = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        total_loss_sum += smooth_l1_loss[i] + rrelu_loss[i] + poisson_loss[i] + center_loss[i];
    }
    total_loss[0] = total_loss_sum / batch_size;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_center);
    cudaFree(d_smooth_l1);
    cudaFree(d_rrelu);
    cudaFree(d_poisson);
    cudaFree(d_center_loss);
}

}  // extern "C"
