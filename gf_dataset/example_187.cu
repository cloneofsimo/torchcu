
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for knowledge distillation loss using bfloat16
__global__ void kd_loss_kernel_bf16(const float* student_output, const float* teacher_output, 
                                    const float* image_gradient, float* loss, int batch_size, int num_classes,
                                    float temperature) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            __nv_bfloat16 s_prob = float_to_bfloat16(expf(student_output[idx * num_classes + i] / temperature));
            __nv_bfloat16 t_prob = float_to_bfloat16(expf(teacher_output[idx * num_classes + i] / temperature));
            
            // Use __hmul for bfloat16 multiplication
            sum += bfloat16_to_float(__hmul(s_prob, logf(bfloat16_to_float(t_prob / s_prob))));
        }

        // Apply image gradient weighting
        loss[idx] = sum * image_gradient[idx]; 
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* student_output = va_arg(args, const float*);
    int student_output_dim0 = va_arg(args, int);
    int student_output_dim1 = va_arg(args, int);

    const float* teacher_output = va_arg(args, const float*);
    int teacher_output_dim0 = va_arg(args, int);
    int teacher_output_dim1 = va_arg(args, int);

    const float* image_gradient = va_arg(args, const float*);
    int image_gradient_dim0 = va_arg(args, int);
    int image_gradient_dim1 = va_arg(args, int);
    int image_gradient_dim2 = va_arg(args, int);

    // Extract output tensor
    float* loss = va_arg(args, float*);

    // Extract temperature
    float temperature = (float)va_arg(args, double);

    va_end(args);

    int batch_size = student_output_dim0;
    int num_classes = student_output_dim1;

    // Allocate device memory
    float *d_student_output, *d_teacher_output, *d_image_gradient, *d_loss;
    cudaMalloc(&d_student_output, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_teacher_output, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_image_gradient, batch_size * image_gradient_dim1 * image_gradient_dim2 * sizeof(float));
    cudaMalloc(&d_loss, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_student_output, student_output, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_teacher_output, teacher_output, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image_gradient, image_gradient, batch_size * image_gradient_dim1 * image_gradient_dim2 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    kd_loss_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_student_output, d_teacher_output, d_image_gradient, d_loss, batch_size, num_classes, temperature
    );

    // Copy result back to host
    cudaMemcpy(loss, d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Reduce the loss (calculate mean)
    float total_loss = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        total_loss += loss[i];
    }
    loss[0] = total_loss / batch_size;

    // Free device memory
    cudaFree(d_student_output);
    cudaFree(d_teacher_output);
    cudaFree(d_image_gradient);
    cudaFree(d_loss);
}

}  // extern "C"
