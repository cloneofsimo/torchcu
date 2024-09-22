
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <math.h>
#include <stdarg.h>

// Function for transposed convolution with ReLU activation
__global__ void conv3d_relu(const float* input, const float* weight, float* output, 
                             int batch_size, int in_channels, int out_channels, 
                             int D_in, int H_in, int W_in, int D_kernel, int H_kernel, int W_kernel, 
                             int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w) {
    
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int out_channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_depth_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_channel_idx < out_channels && out_depth_idx < D_in - D_kernel + 1 + 2 * padding_d) {
        int out_height_idx = out_depth_idx * stride_h + padding_h;
        int out_width_idx = out_depth_idx * stride_w + padding_w;
        
        float sum = 0.0f;
        for (int in_channel_idx = 0; in_channel_idx < in_channels; ++in_channel_idx) {
            for (int kernel_d = 0; kernel_d < D_kernel; ++kernel_d) {
                for (int kernel_h = 0; kernel_h < H_kernel; ++kernel_h) {
                    for (int kernel_w = 0; kernel_w < W_kernel; ++kernel_w) {
                        int in_depth_idx = out_depth_idx + kernel_d - padding_d;
                        int in_height_idx = out_height_idx + kernel_h - padding_h;
                        int in_width_idx = out_width_idx + kernel_w - padding_w;

                        if (in_depth_idx >= 0 && in_depth_idx < D_in && 
                            in_height_idx >= 0 && in_height_idx < H_in &&
                            in_width_idx >= 0 && in_width_idx < W_in) {
                            sum += input[batch_idx * in_channels * D_in * H_in * W_in + 
                                        in_channel_idx * D_in * H_in * W_in +
                                        in_depth_idx * H_in * W_in +
                                        in_height_idx * W_in +
                                        in_width_idx] * 
                                    weight[out_channel_idx * in_channels * D_kernel * H_kernel * W_kernel +
                                            in_channel_idx * D_kernel * H_kernel * W_kernel + 
                                            kernel_d * H_kernel * W_kernel +
                                            kernel_h * W_kernel + 
                                            kernel_w];
                        }
                    }
                }
            }
        }
        output[batch_idx * out_channels * D_in * H_in * W_in + out_channel_idx * D_in * H_in * W_in + out_depth_idx] = fmaxf(sum, 0.0f);
    }
}

// Function for calculating center loss
__global__ void center_loss_kernel(const float* output, const int* labels, float* center_loss, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int label = labels[idx];
        float sum = 0.0f;
        for (int j = 0; j < batch_size; ++j) {
            if (labels[j] == label) {
                sum += output[j * num_classes + label];
            }
        }
        center_loss[label] += powf(output[idx * num_classes + label] - sum / batch_size, 2.0f);
    }
}

// Function for calculating k-th value
__global__ void kth_value_kernel(const float* output, float* kth_value, int batch_size, int num_classes, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float* values = output + idx * num_classes;
        for (int i = 0; i < num_classes - k; ++i) {
            float max_value = values[0];
            int max_index = 0;
            for (int j = 1; j < num_classes; ++j) {
                if (values[j] > max_value) {
                    max_value = values[j];
                    max_index = j;
                }
            }
            values[max_index] = values[num_classes - 1];
            values[num_classes - 1] = max_value;
        }
        kth_value[idx] = values[num_classes - k];
    }
}

// Function for flattening and converting to int8
__global__ void flatten_int8_kernel(const float* output, int8_t* output_int8, int batch_size, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_features) {
        output_int8[idx] = static_cast<int8_t>(output[idx]);
    }
}

extern "C" {

void my_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    int input_tensor_dim3 = va_arg(args, int);
    int input_tensor_dim4 = va_arg(args, int);

    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);
    int weight_dim2 = va_arg(args, int);
    int weight_dim3 = va_arg(args, int);
    int weight_dim4 = va_arg(args, int);

    const int* labels = va_arg(args, const int*);

    int8_t* output_int8 = va_arg(args, int8_t*);
    float* center_loss = va_arg(args, float*);
    float* kth_value = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int in_channels = input_tensor_dim1;
    int D_in = input_tensor_dim2;
    int H_in = input_tensor_dim3;
    int W_in = input_tensor_dim4;

    int out_channels = weight_dim0;
    int D_kernel = weight_dim2;
    int H_kernel = weight_dim3;
    int W_kernel = weight_dim4;

    int num_classes = weight_dim0 * D_in * H_in * W_in;
    int num_features = weight_dim0 * D_in * H_in * W_in;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    int *d_labels;
    int8_t *d_output_int8;
    float *d_center_loss;
    float *d_kth_value;
    
    cudaMalloc(&d_input, batch_size * in_channels * D_in * H_in * W_in * sizeof(float));
    cudaMalloc(&d_weight, out_channels * in_channels * D_kernel * H_kernel * W_kernel * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * D_in * H_in * W_in * sizeof(float));
    cudaMalloc(&d_labels, batch_size * sizeof(int));
    cudaMalloc(&d_output_int8, batch_size * num_features * sizeof(int8_t));
    cudaMalloc(&d_center_loss, num_classes * sizeof(float));
    cudaMalloc(&d_kth_value, batch_size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input_tensor, batch_size * in_channels * D_in * H_in * W_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, out_channels * in_channels * D_kernel * H_kernel * W_kernel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // Perform transposed convolution with ReLU activation
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((D_in - D_kernel + 1 + 2) / 16, (out_channels + 15) / 16, (batch_size + 7) / 8);
    conv3d_relu<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_output, 
                                        batch_size, in_channels, out_channels,
                                        D_in, H_in, W_in, D_kernel, H_kernel, W_kernel,
                                        1, 1, 1, 0, 0, 0);
    
    // Calculate center loss
    threadsPerBlock = dim3(256, 1, 1);
    numBlocks = dim3((batch_size + 255) / 256, 1, 1);
    center_loss_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_labels, d_center_loss, batch_size, num_classes);

    // Calculate k-th value
    threadsPerBlock = dim3(256, 1, 1);
    numBlocks = dim3((batch_size + 255) / 256, 1, 1);
    kth_value_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_kth_value, batch_size, num_classes, 2);

    // Flatten and convert to int8
    threadsPerBlock = dim3(256, 1, 1);
    numBlocks = dim3((batch_size * num_features + 255) / 256, 1, 1);
    flatten_int8_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_output_int8, batch_size, num_features);

    // Copy results back to host
    cudaMemcpy(output_int8, d_output_int8, batch_size * num_features * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_loss, d_center_loss, num_classes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(kth_value, d_kth_value, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaFree(d_labels);
    cudaFree(d_output_int8);
    cudaFree(d_center_loss);
    cudaFree(d_kth_value);
}

} // extern "C"
