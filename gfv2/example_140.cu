
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <stdarg.h>

#define BLOCK_SIZE 16

__global__ void gaussian_blur_3d_kernel(const float* input, float* output, int batch_size, 
                                      int depth, int height, int width, int kernel_size, float sigma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < width && idy < height && idz < depth) {
        float sum = 0.0f;
        int kernel_half = kernel_size / 2;

        for (int kx = -kernel_half; kx <= kernel_half; kx++) {
            for (int ky = -kernel_half; ky <= kernel_half; ky++) {
                for (int kz = -kernel_half; kz <= kernel_half; kz++) {
                    int x = idx + kx;
                    int y = idy + ky;
                    int z = idz + kz;

                    if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) {
                        float kernel_value = exp(-(kx * kx + ky * ky + kz * kz) / (2 * sigma * sigma));
                        sum += input[idz * width * height + idy * width + idx + kx + ky * width + kz * width * height] * kernel_value;
                    }
                }
            }
        }
        output[idz * width * height + idy * width + idx] = sum;
    }
}

__global__ void sobel_kernel_3d(const float* input, float* output, int batch_size, 
                               int depth, int height, int width, int kernel_size, int direction) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < width && idy < height && idz < depth) {
        float sum = 0.0f;
        int kernel_half = kernel_size / 2;

        for (int kx = -kernel_half; kx <= kernel_half; kx++) {
            for (int ky = -kernel_half; ky <= kernel_half; ky++) {
                for (int kz = -kernel_half; kz <= kernel_half; kz++) {
                    int x = idx + kx;
                    int y = idy + ky;
                    int z = idz + kz;

                    if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) {
                        // Sobel kernel for x-direction (replace with other directions if needed)
                        if (direction == 0) { // X
                            if (kx == -1 && ky == -1 && kz == -1) sum -= input[z * width * height + y * width + x];
                            if (kx == -1 && ky == -1 && kz == 0) sum -= 2 * input[z * width * height + y * width + x];
                            if (kx == -1 && ky == -1 && kz == 1) sum -= input[z * width * height + y * width + x];
                            if (kx == -1 && ky == 0 && kz == -1) sum -= input[z * width * height + y * width + x];
                            if (kx == -1 && ky == 0 && kz == 1) sum -= input[z * width * height + y * width + x];
                            if (kx == -1 && ky == 1 && kz == -1) sum -= input[z * width * height + y * width + x];
                            if (kx == -1 && ky == 1 && kz == 0) sum -= 2 * input[z * width * height + y * width + x];
                            if (kx == -1 && ky == 1 && kz == 1) sum -= input[z * width * height + y * width + x];

                            if (kx == 1 && ky == -1 && kz == -1) sum += input[z * width * height + y * width + x];
                            if (kx == 1 && ky == -1 && kz == 0) sum += 2 * input[z * width * height + y * width + x];
                            if (kx == 1 && ky == -1 && kz == 1) sum += input[z * width * height + y * width + x];
                            if (kx == 1 && ky == 0 && kz == -1) sum += input[z * width * height + y * width + x];
                            if (kx == 1 && ky == 0 && kz == 1) sum += input[z * width * height + y * width + x];
                            if (kx == 1 && ky == 1 && kz == -1) sum += input[z * width * height + y * width + x];
                            if (kx == 1 && ky == 1 && kz == 0) sum += 2 * input[z * width * height + y * width + x];
                            if (kx == 1 && ky == 1 && kz == 1) sum += input[z * width * height + y * width + x];
                        } else if (direction == 1) { // Y
                            if (kx == -1 && ky == -1 && kz == -1) sum -= input[z * width * height + y * width + x];
                            if (kx == 0 && ky == -1 && kz == -1) sum -= 2 * input[z * width * height + y * width + x];
                            if (kx == 1 && ky == -1 && kz == -1) sum -= input[z * width * height + y * width + x];
                            if (kx == -1 && ky == -1 && kz == 0) sum -= input[z * width * height + y * width + x];
                            if (kx == 1 && ky == -1 && kz == 0) sum -= input[z * width * height + y * width + x];
                            if (kx == -1 && ky == -1 && kz == 1) sum -= input[z * width * height + y * width + x];
                            if (kx == 0 && ky == -1 && kz == 1) sum -= 2 * input[z * width * height + y * width + x];
                            if (kx == 1 && ky == -1 && kz == 1) sum -= input[z * width * height + y * width + x];

                            if (kx == -1 && ky == 1 && kz == -1) sum += input[z * width * height + y * width + x];
                            if (kx == 0 && ky == 1 && kz == -1) sum += 2 * input[z * width * height + y * width + x];
                            if (kx == 1 && ky == 1 && kz == -1) sum += input[z * width * height + y * width + x];
                            if (kx == -1 && ky == 1 && kz == 0) sum += input[z * width * height + y * width + x];
                            if (kx == 1 && ky == 1 && kz == 0) sum += input[z * width * height + y * width + x];
                            if (kx == -1 && ky == 1 && kz == 1) sum += input[z * width * height + y * width + x];
                            if (kx == 0 && ky == 1 && kz == 1) sum += 2 * input[z * width * height + y * width + x];
                            if (kx == 1 && ky == 1 && kz == 1) sum += input[z * width * height + y * width + x];
                        } else if (direction == 2) { // Z
                            if (kx == -1 && ky == -1 && kz == -1) sum -= input[z * width * height + y * width + x];
                            if (kx == -1 && ky == 0 && kz == -1) sum -= input[z * width * height + y * width + x];
                            if (kx == -1 && ky == 1 && kz == -1) sum -= input[z * width * height + y * width + x];
                            if (kx == 0 && ky == -1 && kz == -1) sum -= input[z * width * height + y * width + x];
                            if (kx == 0 && ky == 1 && kz == -1) sum -= input[z * width * height + y * width + x];
                            if (kx == 1 && ky == -1 && kz == -1) sum -= input[z * width * height + y * width + x];
                            if (kx == 1 && ky == 0 && kz == -1) sum -= input[z * width * height + y * width + x];
                            if (kx == 1 && ky == 1 && kz == -1) sum -= input[z * width * height + y * width + x];

                            if (kx == -1 && ky == -1 && kz == 1) sum += input[z * width * height + y * width + x];
                            if (kx == -1 && ky == 0 && kz == 1) sum += input[z * width * height + y * width + x];
                            if (kx == -1 && ky == 1 && kz == 1) sum += input[z * width * height + y * width + x];
                            if (kx == 0 && ky == -1 && kz == 1) sum += input[z * width * height + y * width + x];
                            if (kx == 0 && ky == 1 && kz == 1) sum += input[z * width * height + y * width + x];
                            if (kx == 1 && ky == -1 && kz == 1) sum += input[z * width * height + y * width + x];
                            if (kx == 1 && ky == 0 && kz == 1) sum += input[z * width * height + y * width + x];
                            if (kx == 1 && ky == 1 && kz == 1) sum += input[z * width * height + y * width + x];
                        }
                    }
                }
            }
        }
        output[idz * width * height + idy * width + idx] = sum;
    }
}

__global__ void gradient_magnitude_3d_kernel(const float* gx, const float* gy, const float* gz, 
                                               float* gradient_magnitude, int batch_size, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < width && idy < height && idz < depth) {
        float gx_val = gx[idz * width * height + idy * width + idx];
        float gy_val = gy[idz * width * height + idy * width + idx];
        float gz_val = gz[idz * width * height + idy * width + idx];

        gradient_magnitude[idz * width * height + idy * width + idx] = sqrtf(gx_val * gx_val + gy_val * gy_val + gz_val * gz_val);
    }
}

__global__ void non_max_suppression_3d_kernel(const float* gradient_magnitude, const int* gradient_direction, 
                                               char* nms_output, int batch_size, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < width && idy < height && idz < depth) {
        float current_magnitude = gradient_magnitude[idz * width * height + idy * width + idx];
        int current_direction = gradient_direction[idz * width * height + idy * width + idx];

        // Check neighbors based on gradient direction
        if (current_direction == 0) { // 0 degrees
            if (current_magnitude <= gradient_magnitude[idz * width * height + idy * width + idx + 1] ||
                current_magnitude <= gradient_magnitude[idz * width * height + (idy + 1) * width + idx]) {
                nms_output[idz * width * height + idy * width + idx] = 0;
                return;
            }
        } else if (current_direction == 1) { // 45 degrees
            if (current_magnitude <= gradient_magnitude[idz * width * height + (idy + 1) * width + idx + 1] ||
                current_magnitude <= gradient_magnitude[idz * width * height + idy * width + idx + 1]) {
                nms_output[idz * width * height + idy * width + idx] = 0;
                return;
            }
        } else if (current_direction == 2) { // 90 degrees
            if (current_magnitude <= gradient_magnitude[idz * width * height + (idy + 1) * width + idx] ||
                current_magnitude <= gradient_magnitude[idz * width * height + idy * width + idx - 1]) {
                nms_output[idz * width * height + idy * width + idx] = 0;
                return;
            }
        } else if (current_direction == 3) { // 135 degrees
            if (current_magnitude <= gradient_magnitude[idz * width * height + (idy + 1) * width + idx - 1] ||
                current_magnitude <= gradient_magnitude[idz * width * height + idy * width + idx - 1]) {
                nms_output[idz * width * height + idy * width + idx] = 0;
                return;
            }
        } else if (current_direction == 4) { // 180 degrees
            if (current_magnitude <= gradient_magnitude[idz * width * height + idy * width + idx - 1] ||
                current_magnitude <= gradient_magnitude[idz * width * height + (idy - 1) * width + idx]) {
                nms_output[idz * width * height + idy * width + idx] = 0;
                return;
            }
        } else if (current_direction == 5) { // 225 degrees
            if (current_magnitude <= gradient_magnitude[idz * width * height + (idy - 1) * width + idx - 1] ||
                current_magnitude <= gradient_magnitude[idz * width * height + idy * width + idx - 1]) {
                nms_output[idz * width * height + idy * width + idx] = 0;
                return;
            }
        } else if (current_direction == 6) { // 270 degrees
            if (current_magnitude <= gradient_magnitude[idz * width * height + (idy - 1) * width + idx] ||
                current_magnitude <= gradient_magnitude[idz * width * height + idy * width + idx + 1]) {
                nms_output[idz * width * height + idy * width + idx] = 0;
                return;
            }
        } else if (current_direction == 7) { // 315 degrees
            if (current_magnitude <= gradient_magnitude[idz * width * height + (idy - 1) * width + idx + 1] ||
                current_magnitude <= gradient_magnitude[idz * width * height + idy * width + idx + 1]) {
                nms_output[idz * width * height + idy * width + idx] = 0;
                return;
            }
        }

        // Suppress if no stronger neighbor found
        nms_output[idz * width * height + idy * width + idx] = 1;
    }
}

__global__ void hysteresis_thresholding_3d_kernel(const char* strong_edges, const char* weak_edges,
                                                   char* connected_edges, int batch_size, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < width && idy < height && idz < depth) {
        if (weak_edges[idz * width * height + idy * width + idx] &&
            (strong_edges[idz * width * height + idy * width + idx + 1] || 
             strong_edges[idz * width * height + (idy + 1) * width + idx] || 
             strong_edges[idz * width * height + (idy - 1) * width + idx] ||
             strong_edges[idz * width * height + idy * width + idx - 1] ||
             strong_edges[(idz + 1) * width * height + idy * width + idx] ||
             strong_edges[(idz - 1) * width * height + idy * width + idx])) {
            connected_edges[idz * width * height + idy * width + idx] = 1;
        } else {
            connected_edges[idz * width * height + idy * width + idx] = strong_edges[idz * width * height + idy * width + idx];
        }
    }
}

extern "C" {

void canny_edge_detector_3d(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int batch_size = va_arg(args, int);
    int depth = va_arg(args, int);
    int height = va_arg(args, int);
    int width = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    char* output = va_arg(args, char*);

    va_end(args);

    // Allocate device memory
    float *d_input, *d_gx, *d_gy, *d_gz, *d_gradient_magnitude;
    char *d_nms_output, *d_connected_edges;
    int *d_gradient_direction; 
    cudaMalloc(&d_input, batch_size * depth * height * width * sizeof(float));
    cudaMalloc(&d_gx, batch_size * depth * height * width * sizeof(float));
    cudaMalloc(&d_gy, batch_size * depth * height * width * sizeof(float));
    cudaMalloc(&d_gz, batch_size * depth * height * width * sizeof(float));
    cudaMalloc(&d_gradient_magnitude, batch_size * depth * height * width * sizeof(float));
    cudaMalloc(&d_gradient_direction, batch_size * depth * height * width * sizeof(int));
    cudaMalloc(&d_nms_output, batch_size * depth * height * width * sizeof(char));
    cudaMalloc(&d_connected_edges, batch_size * depth * height * width * sizeof(char));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * depth * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Gaussian Blur
    dim3 blur_threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 blur_blocks((width + blur_threads.x - 1) / blur_threads.x, 
                    (height + blur_threads.y - 1) / blur_threads.y, 
                    (depth + blur_threads.z - 1) / blur_threads.z);
    gaussian_blur_3d_kernel<<<blur_blocks, blur_threads>>>(d_input, d_input, batch_size, depth, height, width, 3, 1.0f); // 3x3 kernel, sigma 1.0

    // Gradient Calculation (Sobel)
    dim3 sobel_threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 sobel_blocks((width + sobel_threads.x - 1) / sobel_threads.x, 
                    (height + sobel_threads.y - 1) / sobel_threads.y, 
                    (depth + sobel_threads.z - 1) / sobel_threads.z);
    sobel_kernel_3d<<<sobel_blocks, sobel_threads>>>(d_input, d_gx, batch_size, depth, height, width, 3, 0); // X direction
    sobel_kernel_3d<<<sobel_blocks, sobel_threads>>>(d_input, d_gy, batch_size, depth, height, width, 3, 1); // Y direction
    sobel_kernel_3d<<<sobel_blocks, sobel_threads>>>(d_input, d_gz, batch_size, depth, height, width, 3, 2); // Z direction

    // Gradient Magnitude
    dim3 grad_mag_threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grad_mag_blocks((width + grad_mag_threads.x - 1) / grad_mag_threads.x, 
                        (height + grad_mag_threads.y - 1) / grad_mag_threads.y, 
                        (depth + grad_mag_threads.z - 1) / grad_mag_threads.z);
    gradient_magnitude_3d_kernel<<<grad_mag_blocks, grad_mag_threads>>>(d_gx, d_gy, d_gz, d_gradient_magnitude, batch_size, depth, height, width);

    // Gradient Direction
    // Simplified direction calculation (only using x-y plane)
    dim3 dir_threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 dir_blocks((width + dir_threads.x - 1) / dir_threads.x, 
                    (height + dir_threads.y - 1) / dir_threads.y, 
                    (depth + dir_threads.z - 1) / dir_threads.z);
    for (int i = 0; i < batch_size * depth * height * width; i++) {
        float gx_val = d_gx[i];
        float gy_val = d_gy[i];
        if (gx_val == 0.0f && gy_val == 0.0f) {
            d_gradient_direction[i] = 0;
        } else {
            float angle = atan2f(gy_val, gx_val) * 180.0f / M_PI;
            if (angle >= 0 && angle < 22.5) d_gradient_direction[i] = 0;
            else if (angle >= 22.5 && angle < 67.5) d_gradient_direction[i] = 1;
            else if (angle >= 67.5 && angle < 112.5) d_gradient_direction[i] = 2;
            else if (angle >= 112.5 && angle < 157.5) d_gradient_direction[i] = 3;
            else if (angle >= 157.5 && angle < 202.5) d_gradient_direction[i] = 4;
            else if (angle >= 202.5 && angle < 247.5) d_gradient_direction[i] = 5;
            else if (angle >= 247.5 && angle < 292.5) d_gradient_direction[i] = 6;
            else if (angle >= 292.5 && angle < 337.5) d_gradient_direction[i] = 7;
            else d_gradient_direction[i] = 0;
        }
    }

    // Non-Maximum Suppression
    dim3 nms_threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 nms_blocks((width + nms_threads.x - 1) / nms_threads.x, 
                    (height + nms_threads.y - 1) / nms_threads.y, 
                    (depth + nms_threads.z - 1) / nms_threads.z);
    non_max_suppression_3d_kernel<<<nms_blocks, nms_threads>>>(d_gradient_magnitude, d_gradient_direction, d_nms_output, batch_size, depth, height, width);

    // Hysteresis Thresholding
    dim3 hysteresis_threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 hysteresis_blocks((width + hysteresis_threads.x - 1) / hysteresis_threads.x, 
                        (height + hysteresis_threads.y - 1) / hysteresis_threads.y, 
                        (depth + hysteresis_threads.z - 1) / hysteresis_threads.z);
    hysteresis_thresholding_3d_kernel<<<hysteresis_blocks, hysteresis_threads>>>(d_nms_output, d_nms_output, d_connected_edges, batch_size, depth, height, width); // Using d_nms_output as both strong and weak edges here (for simplicity)

    // Copy result back to host
    cudaMemcpy(output, d_connected_edges, batch_size * depth * height * width * sizeof(char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_gz);
    cudaFree(d_gradient_magnitude);
    cudaFree(d_gradient_direction);
    cuda