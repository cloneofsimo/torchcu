
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

__global__ void waveform_analysis_kernel(const float* input_tensor, float* output_tensor,
                                        int input_length, int window_size, int step_size,
                                        int output_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_length) {
        int start = idx * step_size;
        int end = min(start + window_size, input_length);
        float sum = 0.0f;
        float sum_sq = 0.0f;
        float max_val = input_tensor[start];

        for (int i = start; i < end; ++i) {
            sum += input_tensor[i];
            sum_sq += input_tensor[i] * input_tensor[i];
            max_val = fmaxf(max_val, input_tensor[i]);
        }

        output_tensor[idx * 3] = sum / (end - start);
        output_tensor[idx * 3 + 1] = sum_sq / (end - start) - (sum / (end - start)) * (sum / (end - start));
        output_tensor[idx * 3 + 2] = max_val;
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_length = va_arg(args, int);

    int window_size = va_arg(args, int);
    int step_size = va_arg(args, int);

    float* output_tensor = va_arg(args, float*);

    va_end(args);

    int output_length = (input_length - window_size + 1) / step_size;

    float *d_input, *d_output;
    cudaMalloc(&d_input, input_length * sizeof(float));
    cudaMalloc(&d_output, output_length * 3 * sizeof(float));

    cudaMemcpy(d_input, input_tensor, input_length * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 numBlocks((output_length + threadsPerBlock.x - 1) / threadsPerBlock.x);

    waveform_analysis_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output,
                                                              input_length, window_size, step_size,
                                                              output_length);

    cudaMemcpy(output_tensor, d_output, output_length * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

}
