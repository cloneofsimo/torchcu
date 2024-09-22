
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

extern "C" {

__global__ void teacher_student_loss_kernel(const half* teacher_output, const half* student_output, 
                                        const half* teacher_target, const half* student_target,
                                        const float alpha, const float beta, float* loss,
                                        int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels * height * width) {
        float distillation_loss = 0.0f;
        float supervised_loss = 0.0f;

        for (int i = 0; i < channels; ++i) {
            for (int j = 0; j < height; ++j) {
                for (int k = 0; k < width; ++k) {
                    int pos = (idx / (channels * height * width) * channels * height * width + 
                               i * height * width + j * width + k);
                    float t_out = __int2float_rn(teacher_output[pos]);
                    float s_out = __int2float_rn(student_output[pos]);
                    float t_tar = __int2float_rn(teacher_target[pos]);
                    float s_tar = __int2float_rn(student_target[pos]);

                    distillation_loss += (t_out - s_out) * (t_out - s_out);
                    supervised_loss += (s_out - s_tar) * (s_out - s_tar);
                }
            }
        }

        // Normalize losses for a single element
        distillation_loss /= (channels * height * width);
        supervised_loss /= (channels * height * width);

        atomicAdd(loss, alpha * distillation_loss + beta * supervised_loss);
    }
}

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const half* teacher_output = va_arg(args, const half*);
    int teacher_output_dim0 = va_arg(args, int);
    int teacher_output_dim1 = va_arg(args, int);
    int teacher_output_dim2 = va_arg(args, int);
    int teacher_output_dim3 = va_arg(args, int);

    const half* student_output = va_arg(args, const half*);
    int student_output_dim0 = va_arg(args, int);
    int student_output_dim1 = va_arg(args, int);
    int student_output_dim2 = va_arg(args, int);
    int student_output_dim3 = va_arg(args, int);

    const half* teacher_target = va_arg(args, const half*);
    int teacher_target_dim0 = va_arg(args, int);
    int teacher_target_dim1 = va_arg(args, int);
    int teacher_target_dim2 = va_arg(args, int);
    int teacher_target_dim3 = va_arg(args, int);

    const half* student_target = va_arg(args, const half*);
    int student_target_dim0 = va_arg(args, int);
    int student_target_dim1 = va_arg(args, int);
    int student_target_dim2 = va_arg(args, int);
    int student_target_dim3 = va_arg(args, int);

    const float alpha = va_arg(args, float);
    const float beta = va_arg(args, float);

    float* loss = va_arg(args, float*);

    va_end(args);

    int batch_size = teacher_output_dim0;
    int channels = teacher_output_dim1;
    int height = teacher_output_dim2;
    int width = teacher_output_dim3;

    // Allocate device memory
    half *d_teacher_output, *d_student_output, *d_teacher_target, *d_student_target;
    float *d_loss;
    cudaMalloc(&d_teacher_output, batch_size * channels * height * width * sizeof(half));
    cudaMalloc(&d_student_output, batch_size * channels * height * width * sizeof(half));
    cudaMalloc(&d_teacher_target, batch_size * channels * height * width * sizeof(half));
    cudaMalloc(&d_student_target, batch_size * channels * height * width * sizeof(half));
    cudaMalloc(&d_loss, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_teacher_output, teacher_output, batch_size * channels * height * width * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_student_output, student_output, batch_size * channels * height * width * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_teacher_target, teacher_target, batch_size * channels * height * width * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_student_target, student_target, batch_size * channels * height * width * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (batch_size * channels * height * width + threadsPerBlock - 1) / threadsPerBlock;
    teacher_student_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_teacher_output, d_student_output, d_teacher_target, d_student_target, alpha, beta, d_loss,
        batch_size, channels, height, width
    );

    // Copy result back to host
    cudaMemcpy(loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_teacher_output);
    cudaFree(d_student_output);
    cudaFree(d_teacher_target);
    cudaFree(d_student_target);
    cudaFree(d_loss);
}

}  // extern "C"
