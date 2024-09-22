
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

#define BLOCK_SIZE 32 // 32 x 32 threads per block

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void geglu_kernel(const float* x, float* out, int batch_size, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = idx / seq_len / (dim * 2); // batch index
    int sid = (idx / dim / 2) % seq_len; // sequence index
    int did = idx % (dim * 2); // dimension index
    
    if (idx < batch_size * seq_len * (dim * 2)) {
        if (did < dim) {
            out[bid * seq_len * dim + sid * dim + did] = 
                x[bid * seq_len * (dim * 2) + sid * (dim * 2) + did] * 
                sigmoid(x[bid * seq_len * (dim * 2) + sid * (dim * 2) + did + dim]);
        }
    }
}

__global__ void masked_attention_kernel(const float* x, const bool* mask, float* out, 
                                        int batch_size, int seq_len, int dim, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = idx / seq_len / dim; // batch index
    int sid = (idx / dim) % seq_len; // sequence index
    int did = idx % dim; // dimension index
    
    if (idx < batch_size * seq_len * dim) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < n_heads; j++) {
                float q = x[bid * seq_len * dim + i * dim + did + j * head_dim];
                float k = x[bid * seq_len * dim + i * dim + did + j * head_dim + head_dim];
                float v = x[bid * seq_len * dim + i * dim + did + j * head_dim + 2 * head_dim];

                if (mask[bid * seq_len + i] && i <= sid) { 
                    sum += q * k * v * expf(-(q * k) / sqrtf(head_dim));
                }
            }
        }
        out[idx] = sum; 
    }
}

__global__ void layer_norm_kernel(const float* x, float* out, int batch_size, int seq_len, int dim,
                                   float* g, float* b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = idx / seq_len / dim; // batch index
    int sid = (idx / dim) % seq_len; // sequence index
    int did = idx % dim; // dimension index

    if (idx < batch_size * seq_len * dim) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            sum += x[bid * seq_len * dim + i * dim + did];
        }
        float mean = sum / seq_len;

        sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            sum += (x[bid * seq_len * dim + i * dim + did] - mean) * (x[bid * seq_len * dim + i * dim + did] - mean);
        }
        float variance = sum / seq_len;

        out[idx] = g[did] * (x[idx] - mean) / sqrtf(variance + 1e-5) + b[did]; 
    }
}

extern "C" {

void my_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* x = va_arg(args, const float*);
    int x_dim0 = va_arg(args, int);
    int x_dim1 = va_arg(args, int);
    int x_dim2 = va_arg(args, int);

    // Extract mask tensor
    const bool* mask = va_arg(args, const bool*);
    int mask_dim0 = va_arg(args, int);
    int mask_dim1 = va_arg(args, int);
    int mask_dim2 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* out = va_arg(args, float*);

    va_end(args);

    // Allocate device memory
    float *d_x, *d_out, *d_g, *d_b;
    cudaMalloc(&d_x, x_dim0 * x_dim1 * x_dim2 * sizeof(float));
    cudaMalloc(&d_out, x_dim0 * x_dim1 * x_dim2 * sizeof(float));
    cudaMalloc(&d_g, x_dim2 * sizeof(float));
    cudaMalloc(&d_b, x_dim2 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_x, x, x_dim0 * x_dim1 * x_dim2 * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize g and b for LayerNorm
    float* g_h = new float[x_dim2];
    float* b_h = new float[x_dim2];
    for (int i = 0; i < x_dim2; i++) {
        g_h[i] = 1.0f;
        b_h[i] = 0.0f;
    }
    cudaMemcpy(d_g, g_h, x_dim2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_h, x_dim2 * sizeof(float), cudaMemcpyHostToDevice);
    delete[] g_h;
    delete[] b_h;

    // LayerNorm
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((x_dim0 * x_dim1 * x_dim2 + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);
    layer_norm_kernel<<<numBlocks, threadsPerBlock>>>(d_x, d_out, x_dim0, x_dim1, x_dim2, d_g, d_b);

    // GeGLU
    threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    numBlocks = ((x_dim0 * x_dim1 * x_dim2 * 2 + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);
    geglu_kernel<<<numBlocks, threadsPerBlock>>>(d_out, d_x, x_dim0, x_dim1, x_dim2);

    // Masked Attention
    int n_heads = 4;
    int head_dim = x_dim2 / n_heads;
    threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    numBlocks = ((x_dim0 * x_dim1 * x_dim2 + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);
    masked_attention_kernel<<<numBlocks, threadsPerBlock>>>(d_x, mask, d_out, x_dim0, x_dim1, x_dim2, n_heads, head_dim);

    // Add operation
    for (int i = 0; i < x_dim0 * x_dim1 * x_dim2; i++) {
        d_out[i] += d_x[i]; 
    }

    // Copy result back to host
    cudaMemcpy(out, d_out, x_dim0 * x_dim1 * x_dim2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_g);
    cudaFree(d_b);
}

}  // extern "C"
