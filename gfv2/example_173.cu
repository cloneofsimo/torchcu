
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

__global__ void diagonal_kernel(const float* input, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = input[i * (size + 1)];
    }
}

__global__ void bce_loss_kernel(const half* input, const half* target, const half* weights, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float logit = __int2float_rn(__half2float(input[i]));
        float label = __int2float_rn(__half2float(target[i]));
        float weight = __int2float_rn(__half2float(weights[i]));
        output[0] += weight * (-label * logit + (1.0f - label) * log(1.0f + exp(-logit)));
    }
}

__global__ void adaptive_log_softmax_kernel(const half* input, float* output, int size, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int row = i / dim;
        int col = i % dim;
        float max_val = __int2float_rn(__half2float(input[row * dim]));
        for (int j = 1; j < dim; ++j) {
            max_val = fmaxf(max_val, __int2float_rn(__half2float(input[row * dim + j])));
        }
        output[i] = __int2float_rn(__half2float(input[i])) - max_val - log(exp(__int2float_rn(__half2float(input[row * dim])) - max_val) + exp(__int2float_rn(__half2float(input[row * dim + 1])) - max_val) + exp(__int2float_rn(__half2float(input[row * dim + 2])) - max_val) + exp(__int2float_rn(__half2float(input[row * dim + 3])) - max_val));
    }
}

__global__ void nll_loss_kernel(const float* log_softmax, const int* target, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[0] -= log_softmax[i * 4 + target[i]];
    }
}

__global__ void backward_kernel(const half* input, const half* target, const half* weights, float* grad_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float logit = __int2float_rn(__half2float(input[i]));
        float label = __int2float_rn(__half2float(target[i]));
        float weight = __int2float_rn(__half2float(weights[i]));
        grad_output[i] = weight * (-label + 1.0f / (1.0f + exp(-logit)));
    }
}

extern "C" {

void example_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    const int* target = va_arg(args, const int*);
    int target_size = va_arg(args, int);

    const float* weights = va_arg(args, const float*);
    int weights_size = va_arg(args, int);

    // Extract output tensors (assuming pre-allocated)
    float* diagonal = va_arg(args, float*);
    float* bce_loss = va_arg(args, float*);
    float* log_softmax = va_arg(args, float*);
    float* nll_loss = va_arg(args, float*);
    float* grad_output = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    half *d_input, *d_target, *d_weights;
    float *d_diagonal, *d_log_softmax, *d_nll_loss;

    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(half));
    cudaMalloc(&d_target, target_size * sizeof(half));
    cudaMalloc(&d_weights, weights_size * sizeof(half));

    cudaMalloc(&d_diagonal, input_tensor_dim0 * sizeof(float));
    cudaMalloc(&d_log_softmax, input_tensor_dim0 * input_tensor_dim1 * sizeof(float));
    cudaMalloc(&d_nll_loss, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, target_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, weights_size * sizeof(float), cudaMemcpyHostToDevice);

    // Diagonal
    diagonal_kernel<<<(input_tensor_dim0 + 255) / 256, 256>>>(d_input, d_diagonal, input_tensor_dim0);

    // BCE Loss
    bce_loss_kernel<<<(input_tensor_dim0 * input_tensor_dim1 + 255) / 256, 256>>>(d_input, d_target, d_weights, bce_loss, input_tensor_dim0 * input_tensor_dim1);

    // Adaptive Log Softmax
    adaptive_log_softmax_kernel<<<(input_tensor_dim0 * input_tensor_dim1 + 255) / 256, 256>>>(d_input, d_log_softmax, input_tensor_dim0 * input_tensor_dim1, input_tensor_dim1);

    // NLL Loss
    nll_loss_kernel<<<(input_tensor_dim0 + 255) / 256, 256>>>(d_log_softmax, d_target, d_nll_loss, input_tensor_dim0);

    // Backward (BCE)
    backward_kernel<<<(input_tensor_dim0 * input_tensor_dim1 + 255) / 256, 256>>>(d_input, d_target, d_weights, grad_output, input_tensor_dim0 * input_tensor_dim1);

    // Copy results back to host
    cudaMemcpy(diagonal, d_diagonal, input_tensor_dim0 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bce_loss, d_bce_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(log_softmax, d_log_softmax, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nll_loss, d_nll_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_output, grad_output, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_weights);
    cudaFree(d_diagonal);
    cudaFree(d_log_softmax);
    cudaFree(d_nll_loss);
}

}
