
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>

__global__ void softmax_kernel_fp16(const half* input, half* output, int size, float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        float sum = 0.0f;
        for (int j = 0; j < size; ++j) {
            sum += expf((float)input[j] / temperature);
        }
        output[i] = __int2half_rn(expf((float)input[i] / temperature) / sum); 
    }
}

extern "C" {

void hyperparameter_tuned_softmax_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const half* input_tensor = va_arg(args, const half*);
    int input_tensor_size = va_arg(args, int);

    float temperature = (float)va_arg(args, double); // Assuming double for va_arg

    half* output = va_arg(args, half*);

    va_end(args);

    // Allocate device memory for input and output tensors
    half* d_input, *d_output;
    cudaMalloc(&d_input, input_tensor_size * sizeof(half));
    cudaMalloc(&d_output, input_tensor_size * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_size * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel for softmax calculation
    softmax_kernel_fp16<<<(input_tensor_size + 255) / 256, 256>>>(d_input, d_output, input_tensor_size, temperature);

    // Copy result back to host
    cudaMemcpy(output, d_output, input_tensor_size * sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
