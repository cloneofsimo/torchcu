
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end
#include <cutlass/cutlass.h>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// CUDA kernel for NMS
__global__ void nms_kernel(const float* boxes, const float* scores, int* selected_indices,
                           int num_boxes, float iou_threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_boxes) {
        // Check if the box is already suppressed
        if (selected_indices[i] != -1) {
            return;
        }

        for (int j = i + 1; j < num_boxes; ++j) {
            // Check if the box is already suppressed
            if (selected_indices[j] != -1) {
                continue;
            }

            // Calculate IoU
            float iou = calculate_iou(boxes + i * 4, boxes + j * 4);

            // Suppress the box with lower score if IoU is greater than threshold
            if (iou > iou_threshold) {
                if (scores[i] < scores[j]) {
                    selected_indices[i] = -1;
                    break;  // Move to the next box
                } else {
                    selected_indices[j] = -1;
                }
            }
        }
    }
}

// CUDA kernel for PReLU
__global__ void prelu_kernel(const int8_t* input, float prelu_weight, int8_t* output, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_elements) {
        float value = input[i] / 127.0f;  // Scale int8 to [-1, 1]
        value = (value > 0.0f) ? value : value * prelu_weight; // PReLU
        output[i] = __int_as_int8(value * 127.0f);  // Scale back to int8
    }
}

// Function to calculate IoU between two boxes
__device__ float calculate_iou(const float* box1, const float* box2) {
    float x1 = max(box1[0], box2[0]);
    float y1 = max(box1[1], box2[1]);
    float x2 = min(box1[2], box2[2]);
    float y2 = min(box1[3], box2[3]);

    float intersection_area = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
    float box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]);

    return intersection_area / (box1_area + box2_area - intersection_area);
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* boxes = va_arg(args, const float*);
    int boxes_dim0 = va_arg(args, int);
    int boxes_dim1 = va_arg(args, int);
    const float* scores = va_arg(args, const float*);
    int scores_dim0 = va_arg(args, int);
    const float* prelu_weight = va_arg(args, const float*);
    int prelu_weight_dim0 = va_arg(args, int);

    // Allocate device memory for input tensors
    float* d_boxes, *d_scores, *d_prelu_weight;
    cudaMalloc(&d_boxes, boxes_dim0 * boxes_dim1 * sizeof(float));
    cudaMalloc(&d_scores, scores_dim0 * sizeof(float));
    cudaMalloc(&d_prelu_weight, prelu_weight_dim0 * sizeof(float));

    // Copy input tensors to device
    cudaMemcpy(d_boxes, boxes, boxes_dim0 * boxes_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, scores, scores_dim0 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prelu_weight, prelu_weight, prelu_weight_dim0 * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate device memory for NMS output
    int* d_selected_indices;
    cudaMalloc(&d_selected_indices, scores_dim0 * sizeof(int));
    // Initialize selected_indices to -1 (not suppressed)
    cudaMemset(d_selected_indices, -1, scores_dim0 * sizeof(int));

    // Launch NMS kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((scores_dim0 + threadsPerBlock.x - 1) / threadsPerBlock.x);

    nms_kernel<<<numBlocks, threadsPerBlock>>>(d_boxes, d_scores, d_selected_indices, scores_dim0, 0.5f); // Set IOU threshold

    // Allocate device memory for PReLU output
    int8_t* d_prelu_output;
    int selected_count = 0;
    // Count number of selected boxes
    for (int i = 0; i < scores_dim0; ++i) {
        if (d_selected_indices[i] != -1) {
            selected_count++;
        }
    }
    cudaMalloc(&d_prelu_output, selected_count * sizeof(int8_t));

    // Launch PReLU kernel
    threadsPerBlock = 256;
    numBlocks = (selected_count + threadsPerBlock.x - 1) / threadsPerBlock.x;
    prelu_kernel<<<numBlocks, threadsPerBlock>>>(d_scores, d_prelu_weight[0], d_prelu_output, selected_count);

    // Allocate host memory for PReLU output
    int8_t* h_prelu_output = (int8_t*)malloc(selected_count * sizeof(int8_t));

    // Copy PReLU output from device to host
    cudaMemcpy(h_prelu_output, d_prelu_output, selected_count * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Extract output tensor
    int8_t* output = va_arg(args, int8_t*);
    int output_dim0 = va_arg(args, int);
    // Copy PReLU output to output tensor
    memcpy(output, h_prelu_output, selected_count * sizeof(int8_t));

    // Free device memory
    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_prelu_weight);
    cudaFree(d_selected_indices);
    cudaFree(d_prelu_output);

    // Free host memory
    free(h_prelu_output);

    va_end(args);
}

}  // extern "C"
