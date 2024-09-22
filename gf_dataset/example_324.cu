
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp16.h> // For __half
#include <cuda_math.h> // For __int_as_float
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

extern "C" {
#define CUDA_CHECK(x) do { \
        cudaError_t error = x; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void l1_loss_bilinear_fftshift_kernel(const int8_t* input1, const int8_t* input2,
                                                  const float* weight, float* output,
                                                  int batch_size, int height, int width,
                                                  int weight_height, int weight_width) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < batch_size && h < height && w < width) {
        float sum = 0.0f;
        for (int wh = 0; wh < weight_height; wh++) {
            for (int ww = 0; ww < weight_width; ww++) {
                sum += __int_as_float(abs(input1[b * height * width + h * width + w] - input2[b * height * width + h * width + w])) *
                       weight[wh * weight_width * batch_size * height * width + ww * batch_size * height * width + b * height * width + h * width + w];
            }
        }
        output[b * height * width + h * width + w] = sum;
    }
}

__global__ void fftshift_kernel(float* output, int batch_size, int height, int width) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < batch_size && h < height && w < width) {
        int new_h = (h + height / 2) % height;
        int new_w = (w + width / 2) % width;
        output[b * height * width + new_h * width + new_w] = output[b * height * width + h * width + w];
    }
}

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const int8_t* input_tensor1 = va_arg(args, const int8_t*);
    int input_tensor1_dim0 = va_arg(args, int);
    int input_tensor1_dim1 = va_arg(args, int);
    int input_tensor1_dim2 = va_arg(args, int);

    const int8_t* input_tensor2 = va_arg(args, const int8_t*);
    int input_tensor2_dim0 = va_arg(args, int);
    int input_tensor2_dim1 = va_arg(args, int);
    int input_tensor2_dim2 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);
    int weight_dim3 = va_arg(args, int);

    // Extract output tensor
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor1_dim0;
    int height = input_tensor1_dim1;
    int width = input_tensor1_dim2;
    int weight_height = weight_dim0;
    int weight_width = weight_dim1;

    // Allocate device memory
    int8_t *d_input1, *d_input2;
    float *d_weight, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input1, batch_size * height * width * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_input2, batch_size * height * width * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_weight, batch_size * height * width * weight_height * weight_width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * height * width * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input1, input_tensor1, batch_size * height * width * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input2, input_tensor2, batch_size * height * width * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, weight, batch_size * height * width * weight_height * weight_width * sizeof(float), cudaMemcpyHostToDevice));

    // Launch l1 loss and bilinear kernel
    dim3 threadsPerBlock(16, 16, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (width + threadsPerBlock.z - 1) / threadsPerBlock.z);
    l1_loss_bilinear_fftshift_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input1, d_input2, d_weight, d_output,
        batch_size, height, width, weight_height, weight_width
    );

    // Launch fftshift kernel
    fftshift_kernel<<<numBlocks, threadsPerBlock>>>(
        d_output, batch_size, height, width
    );

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, batch_size * height * width * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_input1));
    CUDA_CHECK(cudaFree(d_input2));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));
}
}  // extern "C"
