
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void adaptive_max_pool1d_kernel(const float* input, float* output, int batch_size, int input_size, int output_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int stride = input_size / output_size;
    int start = threadIdx.y * stride;
    int end = min(start + stride, input_size);

    float max_val = input[batch_idx * input_size + start];
    for (int i = start + 1; i < end; ++i) {
        max_val = fmaxf(max_val, input[batch_idx * input_size + i]);
    }

    output[batch_idx * output_size + threadIdx.y] = max_val;
}

extern "C" {

void adaptive_max_pool1d_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    int output_size = va_arg(args, int);

    float* output_tensor = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_size = input_tensor_dim1;

    float* d_input, *d_output;
    cudaMalloc(&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_size * sizeof(float));

    cudaMemcpy(d_input, input_tensor, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);

    adaptive_max_pool1d_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, batch_size, input_size, output_size);

    cudaMemcpy(output_tensor, d_output, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

} // extern "C"
