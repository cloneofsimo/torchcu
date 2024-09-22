
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// Helper function for softshrink
__device__ float softshrink_bf16(float x, float threshold) {
    return (x > threshold) ? (x - threshold) : ((x < -threshold) ? (x + threshold) : 0.0f);
}

// CUDA kernel for adaptive average pooling (1D)
__global__ void adaptive_avg_pool1d_kernel_bf16(const float* input, float* output, 
                                                int batch_size, int input_channels, 
                                                int input_width, int output_width) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch < batch_size && channel < input_channels) {
        float sum = 0.0f;
        for (int i = 0; i < input_width; ++i) {
            __nv_bfloat16 val = float_to_bfloat16(input[(batch * input_channels + channel) * input_width + i]);
            sum += bfloat16_to_float(val);
        }
        output[(batch * input_channels + channel) * output_width] = sum / input_width; 
    }
}

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel_bf16(const float* input, const float* weight, float* output,
                                    int batch_size, int input_dim, int output_dim) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int output_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch < batch_size && output_idx < output_dim) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input[batch * input_dim + i]);
            __nv_bfloat16 b = float_to_bfloat16(weight[output_idx * input_dim + i]);
            sum += bfloat16_to_float(__hmul(a, b));
        }
        output[batch * output_dim + output_idx] = sum;
    }
}

// CUDA kernel for gradient penalty calculation
__global__ void gradient_penalty_kernel_bf16(const float* interpolated_tensor, 
                                                const float* weight, float* grad_norm, 
                                                int batch_size, int input_dim, 
                                                int output_dim, float lambda) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size) {
        float sum_squared = 0.0f;
        for (int output_idx = 0; output_idx < output_dim; ++output_idx) {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; ++i) {
                __nv_bfloat16 a = float_to_bfloat16(interpolated_tensor[batch * input_dim + i]);
                __nv_bfloat16 b = float_to_bfloat16(weight[output_idx * input_dim + i]);
                sum += bfloat16_to_float(__hmul(a, b));
            }
            sum_squared += sum * sum;
        }
        grad_norm[batch] = lambda * (sqrtf(sum_squared) - 1.0f) * (sqrtf(sum_squared) - 1.0f);
    }
}

extern "C" {

void gradient_penalty_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);

    const float* output_tensor = va_arg(args, const float*);
    int output_tensor_dim0 = va_arg(args, int);
    int output_tensor_dim1 = va_arg(args, int);
    int output_tensor_dim2 = va_arg(args, int);
    int output_tensor_dim3 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract lambda
    float lambda = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3; 
    int output_dim = weight_dim0; 

    // Allocate device memory
    float *d_input, *d_output, *d_interpolated, *d_weight, *d_grad_norm;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_interpolated, batch_size * input_dim * sizeof(float)); 
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_grad_norm, batch_size * sizeof(float)); 

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice); 

    // Interpolate between real and fake samples on device
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            float alpha = (rand() / (float)RAND_MAX); 
            d_interpolated[i * input_dim + j] = alpha * d_input[i * input_dim + j] + (1 - alpha) * d_output[i * input_dim + j]; 
        }
    }

    // Adaptive Average Pooling 1D on device
    dim3 threadsPerBlock_pool(16, 16);
    dim3 numBlocks_pool((batch_size + threadsPerBlock_pool.x - 1) / threadsPerBlock_pool.x, 
                       (input_tensor_dim1 + threadsPerBlock_pool.y - 1) / threadsPerBlock_pool.y);

    adaptive_avg_pool1d_kernel_bf16<<<numBlocks_pool, threadsPerBlock_pool>>>(
        d_interpolated, d_interpolated, batch_size, input_tensor_dim1, input_tensor_dim2 * input_tensor_dim3, 1
    );

    // Matrix Multiplication on device
    dim3 threadsPerBlock_matmul(16, 16);
    dim3 numBlocks_matmul((batch_size + threadsPerBlock_matmul.x - 1) / threadsPerBlock_matmul.x, 
                        (output_dim + threadsPerBlock_matmul.y - 1) / threadsPerBlock_matmul.y);

    matmul_kernel_bf16<<<numBlocks_matmul, threadsPerBlock_matmul>>>(
        d_interpolated, d_weight, d_interpolated, batch_size, input_dim, output_dim
    );

    // Calculate gradient penalty on device
    dim3 threadsPerBlock_gp(16, 1);
    dim3 numBlocks_gp((batch_size + threadsPerBlock_gp.x - 1) / threadsPerBlock_gp.x, 1);

    gradient_penalty_kernel_bf16<<<numBlocks_gp, threadsPerBlock_gp>>>(
        d_interpolated, d_weight, d_grad_norm, batch_size, input_dim, output_dim, lambda
    );

    // Reduce gradient penalty on device
    float sum = 0.0f; 
    for (int i = 0; i < batch_size; ++i) {
        sum += d_grad_norm[i];
    } 

    // Copy result back to host
    cudaMemcpy(output, &sum, sizeof(float), cudaMemcpyDeviceToHost); 

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_interpolated); 
    cudaFree(d_weight);
    cudaFree(d_grad_norm); 
}

} // extern "C"
