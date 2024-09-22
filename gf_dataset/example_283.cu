
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/matrix_multiply.h>

extern "C" {

__global__ void supervised_contrastive_loss_kernel_fp16(const half* anchor, const half* positive, 
                                                        const half* negative, float* loss, int batch_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        // Calculate cosine similarity
        float similarity_ap = __fmaf_rn(anchor[idx], positive[idx], -1.0f);
        similarity_ap = cutlass::fast_math::rcp_nr(similarity_ap); 
        similarity_ap = __fmaf_rn(anchor[idx], positive[idx], -1.0f) * similarity_ap; 
        similarity_ap = __fmaf_rn(anchor[idx], positive[idx], -similarity_ap);
        similarity_ap = __fmaf_rn(anchor[idx], positive[idx], -similarity_ap);

        float similarity_an = __fmaf_rn(anchor[idx], negative[idx], -1.0f);
        similarity_an = cutlass::fast_math::rcp_nr(similarity_an); 
        similarity_an = __fmaf_rn(anchor[idx], negative[idx], -1.0f) * similarity_an; 
        similarity_an = __fmaf_rn(anchor[idx], negative[idx], -similarity_an);
        similarity_an = __fmaf_rn(anchor[idx], negative[idx], -similarity_an);
        
        // Apply ReLU and sum up the loss
        float loss_value = __int_as_float( __float2half_rn(1.0f - similarity_ap + similarity_an) > 0 );
        atomicAdd(loss, loss_value);
    }
}

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const half* anchor = va_arg(args, const half*);
    const half* positive = va_arg(args, const half*);
    const half* negative = va_arg(args, const half*);

    // Extract output tensor
    float* loss = va_arg(args, float*);

    int batch_size = va_arg(args, int);

    va_end(args);

    // Allocate device memory
    half* d_anchor, *d_positive, *d_negative;
    cudaMalloc(&d_anchor, batch_size * 128 * sizeof(half));
    cudaMalloc(&d_positive, batch_size * 128 * sizeof(half));
    cudaMalloc(&d_negative, batch_size * 128 * sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_anchor, anchor, batch_size * 128 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positive, positive, batch_size * 128 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative, negative, batch_size * 128 * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    supervised_contrastive_loss_kernel_fp16<<<numBlocks, threadsPerBlock>>>(
        d_anchor, d_positive, d_negative, loss, batch_size
    );

    // Copy result back to host
    cudaMemcpy(loss, loss, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_anchor);
    cudaFree(d_positive);
    cudaFree(d_negative);
}

} // extern "C"
