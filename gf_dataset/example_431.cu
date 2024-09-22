
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // Include for half precision
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <math_functions.h>
#include <stdarg.h>
#include <iostream>

// Helper functions for CutMix calculations
__device__ int __float2int_rn(float val) {
    return static_cast<int>(floorf(val + 0.5f));
}

__device__ __forceinline__ float rand_float(curandState_t* state) {
    return curand_uniform(state);
}

// CUDA kernel for CutMix augmentation
__global__ void cutmix_kernel(const float* input, float* output, const int* target, int* output_target, 
                              int batch_size, int height, int width, float* lam, int* perm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < batch_size) {
        // Calculate cutmix parameters
        float l = lam[i];
        int cut_size = __float2int_rn(sqrtf(1.0f - l) * width);
        
        // Generate random starting points
        int x1 = __float2int_rn(rand_float(&curandState(i)) * (width - cut_size + 1));
        int y1 = __float2int_rn(rand_float(&curandState(i)) * (height - cut_size + 1));
        int x2 = x1 + cut_size;
        int y2 = y1 + cut_size;
        
        // Apply CutMix augmentation
        int perm_idx = perm[i];
        
        for (int c = 0; c < 3; ++c) {
            for (int y = y1; y < y2; ++y) {
                for (int x = x1; x < x2; ++x) {
                    int in_idx = i * height * width * 3 + c * height * width + y * width + x;
                    int out_idx = i * height * width * 3 + c * height * width + y * width + x;
                    output[out_idx] = input[perm_idx * height * width * 3 + c * height * width + y * width + x];
                }
            }
        }
        
        // Update target labels
        output_target[i] = __int2float_rn(l * target[i] + (1 - l) * target[perm_idx]);
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input = va_arg(args, const float*);
    int input_dim0 = va_arg(args, int);
    int input_dim1 = va_arg(args, int);
    int input_dim2 = va_arg(args, int);
    const int* target = va_arg(args, const int*);

    // Extract alpha value
    float alpha = va_arg(args, float);

    // Extract output tensors
    float* output = va_arg(args, float*);
    int* output_target = va_arg(args, int*);

    va_end(args);

    int batch_size = input_dim0;
    int height = input_dim1;
    int width = input_dim2;

    // Allocate device memory
    float* d_input;
    float* d_output;
    int* d_target;
    int* d_output_target;
    float* d_lam;
    int* d_perm;
    cudaMalloc(&d_input, batch_size * height * width * 3 * sizeof(float));
    cudaMalloc(&d_output, batch_size * height * width * 3 * sizeof(float));
    cudaMalloc(&d_target, batch_size * sizeof(int));
    cudaMalloc(&d_output_target, batch_size * sizeof(int));
    cudaMalloc(&d_lam, batch_size * sizeof(float));
    cudaMalloc(&d_perm, batch_size * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * height * width * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // Generate random lam and perm on device
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL32);
    curandSetGeneratorSeed(gen, 12345); // You can set a different seed if desired
    curandGenerateUniform(gen, d_lam, batch_size);
    curandGenerate(gen, d_perm, batch_size, CURAND_DISTRIBUTION_PERMUTE);
    curandDestroyGenerator(gen);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    cutmix_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_target, d_output_target,
                                                    batch_size, height, width, d_lam, d_perm);

    // Copy results back to host
    cudaMemcpy(output, d_output, batch_size * height * width * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_target, d_output_target, batch_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_target);
    cudaFree(d_output_target);
    cudaFree(d_lam);
    cudaFree(d_perm);
}

}  // extern "C"
