
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// CUDA kernel for soft margin loss with global attention
__global__ void soft_margin_loss_with_attention_kernel(const float* input_tensor, 
                                                      const float* attention_weights, 
                                                      float* output,
                                                      int batch_size, 
                                                      int input_size,
                                                      int attention_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * input_size) {
        int b = idx / input_size;
        int i = idx % input_size;

        float attended_value = 0.0f;
        for (int j = 0; j < attention_size; ++j) {
            attended_value += input_tensor[b * input_size + i] * attention_weights[b * attention_size + j];
        }

        output[idx] = -logf(1.0f / (1.0f + expf(-attended_value)));
    }
}

extern "C" {

void soft_margin_loss_with_attention(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract attention weights
    const float* attention_weights = va_arg(args, const float*);
    int attention_weights_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_size = input_tensor_dim1;
    int attention_size = attention_weights_dim0;

    // Allocate device memory
    float *d_input, *d_attention_weights, *d_output;
    cudaMalloc(&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_attention_weights, batch_size * attention_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attention_weights, attention_weights, batch_size * attention_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int num_blocks = (batch_size * input_size + 255) / 256; 
    dim3 threadsPerBlock(256);
    soft_margin_loss_with_attention_kernel<<<num_blocks, threadsPerBlock>>>(
        d_input, d_attention_weights, d_output, batch_size, input_size, attention_size
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_attention_weights);
    cudaFree(d_output);
}

}  // extern "C"
