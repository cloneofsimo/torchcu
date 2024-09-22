
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

extern "C" {

void inverse_fourier_transform_fp32(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    float* output = va_arg(args, float*);

    va_end(args);

    // Input dimensions
    int batch_size = input_tensor_dim0;
    int channels = input_tensor_dim1;
    int dim2 = input_tensor_dim2;
    int dim3 = input_tensor_dim3;

    // Calculate the size of each complex element (real + imaginary)
    int complex_element_size = sizeof(float) * 2;

    // Calculate total input size in bytes
    size_t input_size_bytes = batch_size * channels * dim2 * dim3 * complex_element_size;

    // Allocate device memory for input
    float *d_input;
    cudaMalloc(&d_input, input_size_bytes);

    // Allocate device memory for output
    float *d_output;
    cudaMalloc(&d_output, batch_size * channels * dim2 * dim3 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_size_bytes, cudaMemcpyHostToDevice);

    // Set up cuFFT plan
    cufftHandle plan;
    cufftPlanMany(&plan, 1, &dim3, &dim2, 1, &channels,
                  &dim3, &dim2, 1, &channels, CUFFT_C2R, batch_size);

    // Execute the inverse transform
    cufftExecC2R(plan, (cufftComplex*)d_input, d_output);

    // Destroy the plan
    cufftDestroy(plan);

    // Copy output data back to host
    cudaMemcpy(output, d_output, batch_size * channels * dim2 * dim3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

}  // extern "C"
