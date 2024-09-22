
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>  // For expf
#include <stdarg.h>

// Helper functions for bfloat16 conversion
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// Helper function for calculating IoU
__device__ float calculate_iou(float* box1, float* box2) {
    float x1 = max(box1[0], box2[0]);
    float y1 = max(box1[1], box2[1]);
    float x2 = min(box1[2], box2[2]);
    float y2 = min(box1[3], box2[3]);

    float intersection_area = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
    float box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float union_area = box1_area + box2_area - intersection_area;

    return intersection_area / union_area;
}

// CUDA kernel for NMS with exponential scoring
__global__ void nms_kernel_bf16(const float* boxes, const float* scores, float* keep,
                                 int num_boxes, float iou_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_boxes) {
        if (keep[idx] == 1.0f) {  // Check if box is already suppressed
            for (int j = idx + 1; j < num_boxes; ++j) {
                if (keep[j] == 1.0f) {
                    __nv_bfloat16 score_i = float_to_bfloat16(scores[idx]);
                    __nv_bfloat16 score_j = float_to_bfloat16(scores[j]);
                    __nv_bfloat16 iou = float_to_bfloat16(calculate_iou(boxes + idx * 4, boxes + j * 4));

                    if (iou > iou_threshold && score_j > score_i) {
                        keep[idx] = 0.0f;
                        break;  // Exit the inner loop if the box is suppressed
                    }
                }
            }
        }
    }
}

extern "C" {

void nms_exponential_bf16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* boxes = va_arg(args, const float*);
    int boxes_dim0 = va_arg(args, int);
    int boxes_dim1 = va_arg(args, int);
    const float* scores = va_arg(args, const float*);
    int scores_dim0 = va_arg(args, int);

    // Extract IOU threshold
    float iou_threshold = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* keep = va_arg(args, float*);

    va_end(args);

    int num_boxes = boxes_dim0;

    // Allocate device memory
    float *d_boxes, *d_scores, *d_keep;
    cudaMalloc(&d_boxes, num_boxes * boxes_dim1 * sizeof(float));
    cudaMalloc(&d_scores, num_boxes * sizeof(float));
    cudaMalloc(&d_keep, num_boxes * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_boxes, boxes, num_boxes * boxes_dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, scores, num_boxes * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize 'keep' array to 1.0f on the device
    cudaMemset(d_keep, 1.0f, num_boxes * sizeof(float));

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (num_boxes + threadsPerBlock - 1) / threadsPerBlock;
    nms_kernel_bf16<<<numBlocks, threadsPerBlock>>>(d_boxes, d_scores, d_keep, num_boxes, iou_threshold);

    // Copy result back to host
    cudaMemcpy(keep, d_keep, num_boxes * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_keep);
}

} // extern "C"
