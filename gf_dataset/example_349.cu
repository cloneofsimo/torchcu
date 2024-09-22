
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_traits.h>
#include <cutlass/reduction/threadblock/reduction.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for pairwise Manhattan distance calculation
__global__ void manhattan_distance_kernel(const float* boxes, float* distances, int num_boxes) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_boxes && col < num_boxes) {
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            sum += fabsf(boxes[row * 4 + i] - boxes[col * 4 + i]);
        }
        distances[row * num_boxes + col] = sum;
    }
}

// CUDA kernel for NMS based on distances and scores
__global__ void nms_kernel(const float* distances, const float* scores, bool* keep, int num_boxes, float iou_threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_boxes && keep[i]) {
        for (int j = i + 1; j < num_boxes; ++j) {
            if (keep[j] && distances[i * num_boxes + j] < iou_threshold) {
                if (scores[i] < scores[j]) {
                    keep[i] = false;
                    break;
                } else {
                    keep[j] = false;
                }
            }
        }
    }
}

// CUDA kernel for filtering boxes based on 'keep' array
__global__ void filter_boxes_kernel(const float* boxes, float* output_boxes, const bool* keep, int num_boxes, int* output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_boxes && keep[i]) {
        for (int j = 0; j < 4; ++j) {
            output_boxes[*output_size * 4 + j] = boxes[i * 4 + j];
        }
        (*output_size)++;
    }
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

    float iou_threshold = va_arg(args, float);

    // Extract output tensor (assuming it's preallocated)
    float* output_boxes = va_arg(args, float*);

    va_end(args);

    int num_boxes = boxes_dim0;
    int max_output_size = num_boxes;

    // Allocate device memory
    float* d_boxes, *d_distances, *d_scores;
    bool* d_keep;
    int* d_output_size; 
    cudaMalloc(&d_boxes, num_boxes * 4 * sizeof(float));
    cudaMalloc(&d_distances, num_boxes * num_boxes * sizeof(float));
    cudaMalloc(&d_scores, num_boxes * sizeof(float));
    cudaMalloc(&d_keep, num_boxes * sizeof(bool));
    cudaMalloc(&d_output_size, sizeof(int)); 

    // Copy input data to device
    cudaMemcpy(d_boxes, boxes, num_boxes * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, scores, num_boxes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_size, &max_output_size, sizeof(int), cudaMemcpyHostToDevice);

    // Calculate pairwise Manhattan distances
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_boxes + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_boxes + threadsPerBlock.y - 1) / threadsPerBlock.y);
    manhattan_distance_kernel<<<numBlocks, threadsPerBlock>>>(d_boxes, d_distances, num_boxes);

    // Initialize 'keep' array to all true
    cudaMemset(d_keep, true, num_boxes * sizeof(bool));

    // Perform NMS on the device
    nms_kernel<<<1, 256>>>(d_distances, d_scores, d_keep, num_boxes, iou_threshold);

    // Filter boxes based on 'keep' array
    filter_boxes_kernel<<<1, 256>>>(d_boxes, output_boxes, d_keep, num_boxes, d_output_size);

    // Get the actual output size
    int output_size;
    cudaMemcpy(&output_size, d_output_size, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy result back to host
    cudaMemcpy(output_boxes, output_boxes, output_size * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_boxes);
    cudaFree(d_distances);
    cudaFree(d_scores);
    cudaFree(d_keep);
    cudaFree(d_output_size); 
}

}  // extern "C"

