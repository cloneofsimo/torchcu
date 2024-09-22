
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Structure to represent a bounding box with score
struct Box {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
};

// CUDA kernel for Non-Maximum Suppression (NMS)
__global__ void nms_kernel(const Box* boxes, int num_boxes, float iou_threshold, int* keep, int* num_keep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_boxes) {
        if (keep[idx] == 1) { // Check if box is already suppressed
            for (int i = idx + 1; i < num_boxes; ++i) {
                if (keep[i] == 1) {
                    float x1 = max(boxes[idx].x1, boxes[i].x1);
                    float y1 = max(boxes[idx].y1, boxes[i].y1);
                    float x2 = min(boxes[idx].x2, boxes[i].x2);
                    float y2 = min(boxes[idx].y2, boxes[i].y2);

                    float inter_area = max(0.0f, (x2 - x1)) * max(0.0f, (y2 - y1));
                    float box_area = (boxes[idx].x2 - boxes[idx].x1) * (boxes[idx].y2 - boxes[idx].y1);
                    float other_area = (boxes[i].x2 - boxes[i].x1) * (boxes[i].y2 - boxes[i].y1);
                    float iou = inter_area / (box_area + other_area - inter_area);

                    if (iou > iou_threshold) {
                        keep[i] = 0; // Suppress the box
                        atomicAdd(num_keep, -1); // Decrement the number of kept boxes
                    }
                }
            }
        }
    }
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float* boxes_data = va_arg(args, const float*);
    int boxes_dim0 = va_arg(args, int);
    int boxes_dim1 = va_arg(args, int);

    const float* scores_data = va_arg(args, const float*);
    int scores_dim0 = va_arg(args, int);

    const float* iou_threshold_data = va_arg(args, const float*);
    int iou_threshold_dim0 = va_arg(args, int); 

    // Extract output tensor (assuming it's preallocated)
    int* keep_data = va_arg(args, int*);
    int keep_dim0 = va_arg(args, int);

    va_end(args);

    int num_boxes = boxes_dim0;
    int keep_size = keep_dim0; 

    // Allocate device memory
    Box* d_boxes;
    int* d_keep;
    int* d_num_keep;
    cudaMalloc(&d_boxes, num_boxes * sizeof(Box));
    cudaMalloc(&d_keep, keep_size * sizeof(int));
    cudaMalloc(&d_num_keep, sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_boxes, boxes_data, num_boxes * sizeof(Box), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keep, keep_data, keep_size * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize keep array to 1 (all boxes are initially kept)
    cudaMemset(d_keep, 1, keep_size * sizeof(int));
    
    // Initialize number of kept boxes
    int num_keep = keep_size;
    cudaMemcpy(d_num_keep, &num_keep, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(128);
    dim3 numBlocks((num_boxes + threadsPerBlock.x - 1) / threadsPerBlock.x);
    nms_kernel<<<numBlocks, threadsPerBlock>>>(d_boxes, num_boxes, iou_threshold_data[0], d_keep, d_num_keep);

    // Copy result back to host
    cudaMemcpy(keep_data, d_keep, keep_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_boxes);
    cudaFree(d_keep);
    cudaFree(d_num_keep);
}

}  // extern "C"
