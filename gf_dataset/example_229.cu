
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end
#include <cutlass/cutlass.h>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for NLLLoss using cuDNN
__global__ void nll_loss_kernel(const float* input_tensor, const int* target_tensor, const float* weights, float* output,
                                 int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int target = target_tensor[idx];
        float loss = input_tensor[idx * num_classes + target];
        if (weights != nullptr) {
            loss *= weights[target];
        }
        output[0] -= loss;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    const int* target_tensor = va_arg(args, const int*);
    int target_tensor_dim0 = va_arg(args, int);

    const float* weights = va_arg(args, const float*);
    int weights_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int num_classes = input_tensor_dim1;

    // Allocate device memory
    float *d_input, *d_weights, *d_output;
    int *d_target;
    cudaMalloc(&d_input, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_target, batch_size * sizeof(int));
    cudaMalloc(&d_weights, num_classes * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    if (weights != nullptr) {
        cudaMemcpy(d_weights, weights, num_classes * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        d_weights = nullptr;
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    nll_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_target, d_weights, d_output, batch_size, num_classes
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_weights);
    cudaFree(d_output);
}

}  // extern "C"
