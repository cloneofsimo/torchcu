
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_sm80.h>
#include <cutlass/gemm/device/gemm_universal_sm86.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/matrix_traits.h>
#include <cutlass/reduction/device/reduction.h>
#include <cutlass/reduction/device/reduction_universal.h>
#include <cutlass/reduction/device/reduction_universal_sm80.h>
#include <cutlass/reduction/device/reduction_universal_sm86.h>
#include <cutlass/reduction/reduction.h>
#include <cutlass/tensor_view.h>
#include <cutlass/transform/device/tensor_op.h>
#include <cutlass/transform/device/tensor_op_universal.h>
#include <cutlass/transform/device/tensor_op_universal_sm80.h>
#include <cutlass/transform/device/tensor_op_universal_sm86.h>
#include <cutlass/transform/tensor_op.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/tensor_op.h>
#include <cutlass/util/reference/tensor_op_impl.h>
#include <cutlass/util/reference/tensor_op_sm80.h>
#include <cutlass/util/reference/tensor_op_sm86.h>

#define LOG_SUM_EXP_THRESHOLD 20.0f

__global__ void ctc_loss_kernel(const float* log_probs, const int* targets,
                                 const int* input_lengths, const int* target_lengths,
                                 float* loss, int batch_size, int max_seq_len,
                                 int vocab_size, int blank_index) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    int input_len = input_lengths[batch_idx];
    int target_len = target_lengths[batch_idx];

    float sum_log_probs = 0.0f;

    int t = 0;
    int s = 0;
    int prev_target = -1;

    // Iterate over input sequence
    while (t < input_len) {
        int target = targets[batch_idx * max_seq_len + s];
        
        if (target != blank_index && target != prev_target) {
            // Match target with log probability
            sum_log_probs += log_probs[batch_idx * max_seq_len * vocab_size + t * vocab_size + target];
            prev_target = target;
            s++;
        }
        t++;
    }

    // Handle remaining targets (if any)
    while (s < target_len) {
        int target = targets[batch_idx * max_seq_len + s];

        if (target != blank_index && target != prev_target) {
            sum_log_probs += log_probs[batch_idx * max_seq_len * vocab_size + (input_len - 1) * vocab_size + target];
            prev_target = target;
            s++;
        }
    }

    // Calculate negative log probability
    loss[batch_idx] = -sum_log_probs;
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract arguments
    const float* log_probs = va_arg(args, const float*);
    int log_probs_dim0 = va_arg(args, int);
    int log_probs_dim1 = va_arg(args, int);
    int log_probs_dim2 = va_arg(args, int);

    const int* targets = va_arg(args, const int*);
    int targets_dim0 = va_arg(args, int);
    int targets_dim1 = va_arg(args, int);

    const int* input_lengths = va_arg(args, const int*);
    int input_lengths_dim0 = va_arg(args, int);

    const int* target_lengths = va_arg(args, const int*);
    int target_lengths_dim0 = va_arg(args, int);

    float* loss = va_arg(args, float*);

    va_end(args);

    int batch_size = log_probs_dim0;
    int max_seq_len = log_probs_dim1;
    int vocab_size = log_probs_dim2;
    int blank_index = 0;

    // Allocate device memory
    float *d_log_probs, *d_loss;
    int *d_targets, *d_input_lengths, *d_target_lengths;
    cudaMalloc(&d_log_probs, batch_size * max_seq_len * vocab_size * sizeof(float));
    cudaMalloc(&d_targets, batch_size * targets_dim1 * sizeof(int));
    cudaMalloc(&d_input_lengths, input_lengths_dim0 * sizeof(int));
    cudaMalloc(&d_target_lengths, target_lengths_dim0 * sizeof(int));
    cudaMalloc(&d_loss, batch_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_log_probs, log_probs, batch_size * max_seq_len * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets, batch_size * targets_dim1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_lengths, input_lengths, input_lengths_dim0 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_lengths, target_lengths, target_lengths_dim0 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    ctc_loss_kernel<<<numBlocks, threadsPerBlock>>>(
        d_log_probs, d_targets, d_input_lengths, d_target_lengths, d_loss,
        batch_size, max_seq_len, vocab_size, blank_index
    );

    // Copy result back to host
    cudaMemcpy(loss, d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_log_probs);
    cudaFree(d_targets);
    cudaFree(d_input_lengths);
    cudaFree(d_target_lengths);
    cudaFree(d_loss);
}
}  // extern "C"
