
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void elementwise_diff_kernel(float* input1, const float* input2, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        input1[i] -= input2[i];
    }
}

extern "C" {
    void torch_function(int num_args, ...) {
        va_list args;
        va_start(args, num_args);

        // Extract input tensors
        float* input1 = va_arg(args, float*);
        int input1_dim0 = va_arg(args, int);
        int input1_dim1 = va_arg(args, int);

        const float* input2 = va_arg(args, const float*);
        int input2_dim0 = va_arg(args, int);
        int input2_dim1 = va_arg(args, int);

        va_end(args);

        int size = input1_dim0 * input1_dim1;

        // Launch kernel
        dim3 threadsPerBlock(256);
        dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);
        elementwise_diff_kernel<<<numBlocks, threadsPerBlock>>>(input1, input2, size);
    }
}
