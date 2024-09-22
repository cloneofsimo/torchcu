
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

// Helper function for adaptive log softmax
__device__ __forceinline__ float exp_f16_to_float(__half h) {
    return __int_as_float(static_cast<int>(h));
}

// CUDA kernel for teacher-student loss calculation with adaptive log softmax
__global__ void teacher_student_loss_kernel_fp16(
    const half* student_output, const half* teacher_output, float temperature,
    float* loss, int batch_size, int num_classes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += exp_f16_to_float(student_output[i * num_classes + j]);
        }
        float loss_value = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            float student_prob = exp_f16_to_float(student_output[i * num_classes + j]) / sum_exp;
            float teacher_prob = exp_f16_to_float(teacher_output[i * num_classes + j]) / temperature;
            loss_value -= teacher_prob * logf(student_prob);
        }
        loss[i] = loss_value;
    }
}

extern "C" {

void teacher_student_loss_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const half* student_output = va_arg(args, const half*);
    int student_output_dim0 = va_arg(args, int);
    int student_output_dim1 = va_arg(args, int);

    const half* teacher_output = va_arg(args, const half*);
    int teacher_output_dim0 = va_arg(args, int);
    int teacher_output_dim1 = va_arg(args, int);

    // Extract temperature
    float temperature = va_arg(args, float);

    // Extract output tensor
    float* loss = va_arg(args, float*);

    va_end(args);

    int batch_size = student_output_dim0;
    int num_classes = student_output_dim1;

    // Allocate device memory
    half *d_student_output, *d_teacher_output;
    float *d_loss;
    cudaMalloc(&d_student_output, batch_size * num_classes * sizeof(half));
    cudaMalloc(&d_teacher_output, batch_size * num_classes * sizeof(half));
    cudaMalloc(&d_loss, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_student_output, student_output, batch_size * num_classes * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_teacher_output, teacher_output, batch_size * num_classes * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    teacher_student_loss_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_student_output, d_teacher_output, temperature, d_loss, batch_size, num_classes
    );

    // Copy result back to host
    cudaMemcpy(loss, d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_student_output);
    cudaFree(d_teacher_output);
    cudaFree(d_loss);
}

}  // extern "C"
