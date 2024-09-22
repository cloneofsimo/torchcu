
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void logsumexp_loss_kernel(const float* input, const float* target, float* output, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float lse = -INFINITY;
        for (int i = 0; i < num_classes; ++i) {
            lse = fmaxf(lse, input[idx * num_classes + i]);
        }
        float loss = lse;
        for (int i = 0; i < num_classes; ++i) {
            loss -= input[idx * num_classes + i] * target[idx * num_classes + i];
        }
        output[idx] = loss;
    }
}

extern "C" {

void logsumexp_loss_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);

    const float* target = va_arg(args, const float*);
    int target_dim0 = va_arg(args, int);
    int target_dim1 = va_arg(args, int);

    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_dim0;
    int num_classes = input_dim1;

    float *d_input, *d_target, *d_output;
    cudaMalloc(&d_input, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_target, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    cudaMemcpy(d_input, input, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    logsumexp_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_target, d_output, batch_size, num_classes
    );

    cudaMemcpy(output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_output);
}

}  // extern "C"
