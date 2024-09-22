
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h> 

// Helper function for upsampling with linear interpolation
__global__ void upsample_linear_kernel(const float* input, float* output, int batch_size, int num_channels, int num_frames, int upsampling_factor) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if (frame < num_frames * upsampling_factor && channel < num_channels && batch < batch_size) {
        int original_frame = frame / upsampling_factor;
        float weight = (frame % upsampling_factor) / (float) upsampling_factor;

        // Linear interpolation: output = (1 - weight) * input[original_frame] + weight * input[original_frame + 1]
        output[batch * num_channels * num_frames * upsampling_factor + channel * num_frames * upsampling_factor + frame] =
            (1.0f - weight) * input[batch * num_channels * num_frames + channel * num_frames + original_frame] +
            weight * input[batch * num_channels * num_frames + channel * num_frames + (original_frame + 1)]; 
    }
}

// CUDA kernel for binary cross-entropy loss
__global__ void bce_loss_kernel(const float* input, const float* target, float* output, int batch_size, int num_channels, int num_frames) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < batch_size * num_channels * num_frames) {
        float input_value = input[index];
        float target_value = target[index];

        // Clamp input values to avoid numerical instability
        input_value = fmaxf(input_value, 1e-7f); 
        input_value = fminf(input_value, 1.0f - 1e-7f); 

        output[0] += -target_value * logf(input_value) - (1.0f - target_value) * logf(1.0f - input_value);
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors and upsampling factor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);

    const float* target_tensor = va_arg(args, const float*);
    int target_tensor_dim0 = va_arg(args, int);
    int target_tensor_dim1 = va_arg(args, int);
    int target_tensor_dim2 = va_arg(args, int);

    int upsampling_factor = va_arg(args, int);

    // Extract output tensor
    float* output_tensor = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_target, *d_output;
    cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * sizeof(float));
    cudaMalloc(&d_target, target_tensor_dim0 * target_tensor_dim1 * target_tensor_dim2 * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, target_tensor_dim0 * target_tensor_dim1 * target_tensor_dim2 * sizeof(float), cudaMemcpyHostToDevice);

    // Upsample audio input
    int upsampled_num_frames = input_tensor_dim2 * upsampling_factor;
    float *d_upsampled_output;
    cudaMalloc(&d_upsampled_output, input_tensor_dim0 * input_tensor_dim1 * upsampled_num_frames * sizeof(float));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((upsampled_num_frames + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (input_tensor_dim1 + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (input_tensor_dim0 + threadsPerBlock.z - 1) / threadsPerBlock.z);

    upsample_linear_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_upsampled_output, input_tensor_dim0, input_tensor_dim1, input_tensor_dim2, upsampling_factor
    );

    // Calculate binary cross-entropy loss
    dim3 bce_threadsPerBlock(1024);
    dim3 bce_numBlocks((input_tensor_dim0 * input_tensor_dim1 * upsampled_num_frames + bce_threadsPerBlock.x - 1) / bce_threadsPerBlock.x);

    bce_loss_kernel<<<bce_numBlocks, bce_threadsPerBlock>>>(
        d_upsampled_output, d_target, d_output, input_tensor_dim0, input_tensor_dim1, upsampled_num_frames
    );

    // Copy result back to host
    cudaMemcpy(output_tensor, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_output);
    cudaFree(d_upsampled_output);
}

}  // extern "C"
