
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for Hadamard product, Chebyshev distance, and sigmoid focal loss
__global__ void hadamard_chebyshev_focal_kernel_bf16(const float* input1, const float* input2, float* output,
                                                int batch_size, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        // Hadamard product
        for (int i = 0; i < input_size; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input1[idx * input_size + i]);
            __nv_bfloat16 b = float_to_bfloat16(input2[idx * input_size + i]);
            output[idx * 3 + 0] = bfloat16_to_float(__hmul(a, b)); // Store hadamard product
        }

        // Chebyshev distance
        float max_diff = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input1[idx * input_size + i]);
            __nv_bfloat16 b = float_to_bfloat16(input2[idx * input_size + i]);
            max_diff = fmaxf(max_diff, fabsf(bfloat16_to_float(a - b)));
        }
        output[idx * 3 + 1] = max_diff; // Store Chebyshev distance

        // Sigmoid focal loss
        float loss = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input1[idx * input_size + i]);
            __nv_bfloat16 b = float_to_bfloat16(input2[idx * input_size + i]);
            float sigmoid_loss = -bfloat16_to_float(b) * logf(bfloat16_to_float(a)) - (1.0f - bfloat16_to_float(b)) * logf(1.0f - bfloat16_to_float(a));
            loss += sigmoid_loss * (1.0f - bfloat16_to_float(b)) * (1.0f - bfloat16_to_float(b));
        }
        output[idx * 3 + 2] = loss; // Store sigmoid focal loss
    }
}

extern "C" {

void forward(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input1 = va_arg(args, const float*);
    int input1_dim0 = va_arg(args, int);
    int input1_dim1 = va_arg(args, int);

    const float* input2 = va_arg(args, const float*);
    int input2_dim0 = va_arg(args, int);
    int input2_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input1_dim0;
    int input_size = input1_dim1;

    // Allocate device memory
    float *d_input1, *d_input2, *d_output;
    cudaMalloc(&d_input1, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_input2, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * 3 * sizeof(float)); // 3 for hadamard, chebyshev, focal

    // Copy input data to device
    cudaMemcpy(d_input1, input1, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256; // Choose an appropriate value for threadsPerBlock
    int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    hadamard_chebyshev_focal_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input1, d_input2, d_output, batch_size, input_size
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}

}  // extern "C"
