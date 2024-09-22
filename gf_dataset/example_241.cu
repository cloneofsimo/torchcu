
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for finding the top-k values and indices in an image tensor
__global__ void image_topk_kernel_bf16(const __nv_bfloat16* image_bf16, int k, __nv_bfloat16* topk_values_bf16, int* topk_indices,
                                      int batch_size, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < batch_size && row < height) {
        // Calculate the linear index of the current pixel
        int pixel_index = idx * height * width + row * width;

        // Find top-k values and indices for the current pixel
        __nv_bfloat16 topk_values_local[WARP_SIZE];
        int topk_indices_local[WARP_SIZE];

        // Initialize the top-k values and indices
        for (int i = 0; i < WARP_SIZE; ++i) {
            topk_values_local[i] = -INFINITY;
            topk_indices_local[i] = -1;
        }

        // Iterate over all pixels in the current row
        for (int col = 0; col < width; ++col) {
            int current_index = pixel_index + col;
            __nv_bfloat16 current_value = image_bf16[current_index];

            // Find the minimum value in the top-k values
            __nv_bfloat16 min_value = topk_values_local[0];
            int min_index = 0;
            for (int i = 1; i < k; ++i) {
                if (topk_values_local[i] < min_value) {
                    min_value = topk_values_local[i];
                    min_index = i;
                }
            }

            // If the current value is greater than the minimum value, replace the minimum value
            if (current_value > min_value) {
                topk_values_local[min_index] = current_value;
                topk_indices_local[min_index] = col;
            }
        }

        // Store the top-k values and indices
        for (int i = 0; i < k; ++i) {
            topk_values_bf16[idx * k + i] = topk_values_local[i];
            topk_indices[idx * k + i] = row * width + topk_indices_local[i];
        }
    }
}

// CUDA kernel for computing the image gradient
__global__ void image_gradient_kernel(const __nv_bfloat16* topk_values_bf16, const int* topk_indices, float* image_gradient,
                                        int batch_size, int height, int width, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < batch_size && row < height) {
        // Calculate the linear index of the current pixel
        int pixel_index = idx * height * width + row * width;

        // Iterate over the top-k indices
        for (int i = 0; i < k; ++i) {
            int topk_index = topk_indices[idx * k + i];

            // If the current pixel is a top-k pixel, set its gradient to its top-k value
            if (topk_index == pixel_index) {
                image_gradient[pixel_index] = bfloat16_to_float(topk_values_bf16[idx * k + i]);
            }
        }
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input image tensor
    const float* input_image = va_arg(args, const float*);
    int input_image_dim0 = va_arg(args, int);
    int input_image_dim1 = va_arg(args, int);
    int input_image_dim2 = va_arg(args, int);

    // Extract k
    int k = va_arg(args, int);

    // Extract output tensors
    float* output_gradient = va_arg(args, float*);
    int* output_indices = va_arg(args, int*);

    va_end(args);

    int batch_size = input_image_dim0;
    int height = input_image_dim1;
    int width = input_image_dim2;

    // Allocate device memory
    __nv_bfloat16 *d_image_bf16, *d_topk_values_bf16;
    int *d_topk_indices;
    float *d_image_gradient;
    cudaMalloc(&d_image_bf16, batch_size * height * width * sizeof(__nv_bfloat16));
    cudaMalloc(&d_topk_values_bf16, batch_size * k * sizeof(__nv_bfloat16));
    cudaMalloc(&d_topk_indices, batch_size * k * sizeof(int));
    cudaMalloc(&d_image_gradient, batch_size * height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_image_bf16, input_image, batch_size * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch top-k kernel
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    image_topk_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_image_bf16, k, d_topk_values_bf16, d_topk_indices, batch_size, height, width
    );

    // Launch image gradient kernel
    image_gradient_kernel<<<numBlocks, threadsPerBlock>>>(
        d_topk_values_bf16, d_topk_indices, d_image_gradient, batch_size, height, width, k
    );

    // Copy results back to host
    cudaMemcpy(output_gradient, d_image_gradient, batch_size * height * width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_indices, d_topk_indices, batch_size * k * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image_bf16);
    cudaFree(d_topk_values_bf16);
    cudaFree(d_topk_indices);
    cudaFree(d_image_gradient);
}

}  // extern "C"
