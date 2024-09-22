
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cufft.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

__global__ void conv_fft_bf16_kernel(const float* input_tensor, const float* weight, float* output, 
                                      int batch_size, int in_channels, int out_channels, int input_size, int kernel_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch_size && channel_idx < out_channels) {
        // Calculate output size
        int output_size = input_size + kernel_size - 1;
        int padded_size = output_size;

        // Allocate device memory for FFT
        cufftComplex* d_input_fft;
        cufftComplex* d_weight_fft;
        cufftComplex* d_output_fft;
        cudaMalloc(&d_input_fft, padded_size * sizeof(cufftComplex));
        cudaMalloc(&d_weight_fft, kernel_size * sizeof(cufftComplex));
        cudaMalloc(&d_output_fft, padded_size * sizeof(cufftComplex));

        // Load input and weight data into device memory
        cudaMemcpy(d_input_fft, &input_tensor[batch_idx * in_channels * input_size + channel_idx * input_size],
                   input_size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight_fft, &weight[channel_idx * in_channels * kernel_size],
                   kernel_size * sizeof(cufftComplex), cudaMemcpyHostToDevice);

        // Perform FFT on input and weight
        cufftHandle plan;
        cufftPlan1d(&plan, padded_size, CUFFT_C2C, 1); 
        cufftExecC2C(plan, d_input_fft, d_input_fft, CUFFT_FORWARD);
        cufftExecC2C(plan, d_weight_fft, d_weight_fft, CUFFT_FORWARD);
        cufftDestroy(plan);

        // Frequency-domain multiplication
        for (int i = 0; i < padded_size; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(d_input_fft[i].x);
            __nv_bfloat16 b = float_to_bfloat16(d_weight_fft[i].x);
            d_output_fft[i].x = bfloat16_to_float(__hmul(a, b));
            d_output_fft[i].y = 0.0f; // Assuming real-valued convolution
        }

        // Inverse FFT
        cufftPlan1d(&plan, padded_size, CUFFT_C2C, 1);
        cufftExecC2C(plan, d_output_fft, d_output_fft, CUFFT_INVERSE);
        cufftDestroy(plan);

        // Copy output back to host memory
        cudaMemcpy(&output[batch_idx * out_channels * output_size + channel_idx * output_size], d_output_fft,
                   output_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input_fft);
        cudaFree(d_weight_fft);
        cudaFree(d_output_fft);
    }
}

extern "C" {

void conv_fft_bfloat16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int out_channels = weight_dim0;
    int input_size = input_tensor_dim2;
    int kernel_size = weight_dim2;

    // Calculate output size
    int output_size = input_size + kernel_size - 1;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, batch_size * in_channels * input_size * sizeof(float));
    cudaMalloc(&d_weight, out_channels * in_channels * kernel_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * output_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * in_channels * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, out_channels * in_channels * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (out_channels + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conv_fft_bf16_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_output,
                                                     batch_size, in_channels, out_channels, input_size, kernel_size);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * out_channels * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

} // extern "C"
