#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

extern "C" {

__global__ void addBiasKernel(float* output, const float* bias, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < m && idy < n) {
        output[idx * n + idy] += bias[idy];
    }
}

void linearCUDA(const float* x, const float* w, const float* b, float* y, int batch_size, int seq_len, int in_features, int out_features) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float* d_x, *d_w, *d_b, *d_y;
    size_t x_size = batch_size * seq_len * in_features * sizeof(float);
    size_t w_size = out_features * in_features * sizeof(float);
    size_t b_size = out_features * sizeof(float);
    size_t y_size = batch_size * seq_len * out_features * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_x, x_size));
    CHECK_CUDA(cudaMalloc(&d_w, w_size));
    CHECK_CUDA(cudaMalloc(&d_b, b_size));
    CHECK_CUDA(cudaMalloc(&d_y, y_size));

    CHECK_CUDA(cudaMemcpy(d_x, x, x_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, w, w_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    int m = batch_size * seq_len;
    int k = in_features;
    int n = out_features;

    // Corrected cublasSgemm call
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             n, m, k,
                             &alpha,
                             d_w, k,
                             d_x, k,
                             &beta,
                             d_y, n));

    dim3 blockDim(16, 16);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
    addBiasKernel<<<gridDim, blockDim>>>(d_y, d_b, m, n);

    CHECK_CUDA(cudaMemcpy(y, d_y, y_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_y));

    CHECK_CUBLAS(cublasDestroy(handle));
}

void linear(
    const int num_args,
    const float* x, int x_dim0, int x_dim1, int x_dim2,
    const float* w, int w_dim0, int w_dim1,
    const float* b, int b_dim0,
    float* y
) {
    linearCUDA(x, w, b, y, x_dim0, x_dim1, x_dim2, w_dim0);
}

}  // extern "C"


int main() {
    const int batch_size = 2;
    const int seq_len = 4;
    const int in_features = 5;
    const int out_features = 3;

    float x[batch_size * seq_len * in_features];
    float w[out_features * in_features];
    float b[out_features];
    float y[batch_size * seq_len * out_features];

    // Initialize input data (you can replace this with your own initialization)
    for (int i = 0; i < batch_size * seq_len * in_features; ++i) x[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < out_features * in_features; ++i) w[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < out_features; ++i) b[i] = static_cast<float>(rand()) / RAND_MAX;

    linearCUDA(x, w, b, y, batch_size, seq_len, in_features, out_features);

    // Print the result (you can modify this part as needed)
    std::cout << "Output:" << std::endl;
    for (int i = 0; i < batch_size * seq_len; ++i) {
        for (int j = 0; j < out_features; ++j) {
            std::cout << y[i * out_features + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
