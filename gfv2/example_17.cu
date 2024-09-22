
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/epilogue/threadblock/linear_combination.h>
#include <cutlass/epilogue/threadblock/scale_and_add.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/transform/threadblock/predicated_tile_store.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/tensor_view.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <stdarg.h>

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// Structure to hold CUDA kernel arguments
struct CosineEmbeddingLossArgs {
    const float *input1;
    const float *input2;
    const int *labels;
    float *output;
    int batch_size;
    int embedding_dim;
};

// CUDA kernel for cosine similarity calculation
template <typename T>
__global__ void cosine_similarity_kernel(const T *input1, const T *input2, T *cosine_sim, int batch_size, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        T dot_product = 0;
        for (int i = 0; i < embedding_dim; i++) {
            dot_product += input1[idx * embedding_dim + i] * input2[idx * embedding_dim + i];
        }
        T norm1 = 0, norm2 = 0;
        for (int i = 0; i < embedding_dim; i++) {
            norm1 += input1[idx * embedding_dim + i] * input1[idx * embedding_dim + i];
            norm2 += input2[idx * embedding_dim + i] * input2[idx * embedding_dim + i];
        }
        cosine_sim[idx] = dot_product / (sqrt(norm1) * sqrt(norm2));
    }
}

// CUDA kernel for RMSE normalization
template <typename T>
__global__ void rmse_normalization_kernel(const T *input, T *output, int batch_size, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        T sum_squares = 0;
        for (int i = 0; i < embedding_dim; i++) {
            sum_squares += input[idx * embedding_dim + i] * input[idx * embedding_dim + i];
        }
        T rmse = sqrt(sum_squares / embedding_dim);
        for (int i = 0; i < embedding_dim; i++) {
            output[idx * embedding_dim + i] = input[idx * embedding_dim + i] / rmse;
        }
    }
}

// CUDA kernel for cosine embedding loss calculation
template <typename T>
__global__ void cosine_embedding_loss_kernel(const T *cosine_sim, const int *labels, T *loss, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        T margin = 1.0; // Default margin for cosine embedding loss
        T target = labels[idx] == 1 ? 1.0 : -1.0;
        loss[idx] = max(T(0), margin - target * cosine_sim[idx]);
    }
}

// Helper function for allocating CUDA memory
template <typename T>
T *allocate_device_memory(int size) {
    T *device_memory;
    cudaMalloc((void **)&device_memory, size * sizeof(T));
    if (device_memory == nullptr) {
        throw std::runtime_error("Error allocating CUDA memory.");
    }
    return device_memory;
}

// Helper function for copying data to CUDA device
template <typename T>
void copy_to_device(const T *host_data, T *device_data, int size) {
    cudaMemcpy(device_data, host_data, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize(); // Ensure data is copied before proceeding
}

// Helper function for copying data from CUDA device
template <typename T>
void copy_from_device(T *device_data, T *host_data, int size) {
    cudaMemcpy(host_data, device_data, size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); // Ensure data is copied before proceeding
}

// CUDA kernel for cosine embedding loss with RMSE normalization
template <typename T>
__global__ void cosine_embedding_loss_rms_kernel(const T *input1, const T *input2, const int *labels, T *loss, 
                                            int batch_size, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        // Calculate RMSE normalization
        T rmse1 = 0, rmse2 = 0;
        for (int i = 0; i < embedding_dim; i++) {
            rmse1 += input1[idx * embedding_dim + i] * input1[idx * embedding_dim + i];
            rmse2 += input2[idx * embedding_dim + i] * input2[idx * embedding_dim + i];
        }
        rmse1 = sqrt(rmse1 / embedding_dim);
        rmse2 = sqrt(rmse2 / embedding_dim);

        // Normalize inputs
        T dot_product = 0;
        for (int i = 0; i < embedding_dim; i++) {
            dot_product += (input1[idx * embedding_dim + i] / rmse1) * (input2[idx * embedding_dim + i] / rmse2);
        }
        T norm1 = 0, norm2 = 0;
        for (int i = 0; i < embedding_dim; i++) {
            norm1 += (input1[idx * embedding_dim + i] / rmse1) * (input1[idx * embedding_dim + i] / rmse1);
            norm2 += (input2[idx * embedding_dim + i] / rmse2) * (input2[idx * embedding_dim + i] / rmse2);
        }

        // Calculate cosine similarity
        T cosine_sim = dot_product / (sqrt(norm1) * sqrt(norm2));

        // Calculate cosine embedding loss
        T margin = 1.0;
        T target = labels[idx] == 1 ? 1.0 : -1.0;
        loss[idx] = max(T(0), margin - target * cosine_sim);
    }
}

// Specialized version for half-precision with Cutlass
template <>
__global__ void cosine_embedding_loss_rms_kernel<half>(const half *input1, const half *input2, const int *labels, 
                                                        half *loss, int batch_size, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        // Allocate shared memory for input tensors and normalization factors
        __shared__ half shared_input1[128];
        __shared__ half shared_input2[128];
        __shared__ half shared_rmse1, shared_rmse2;

        // Load input data into shared memory
        if (threadIdx.x < embedding_dim) {
            shared_input1[threadIdx.x] = input1[idx * embedding_dim + threadIdx.x];
            shared_input2[threadIdx.x] = input2[idx * embedding_dim + threadIdx.x];
        }

        // Compute RMSE normalization using shared memory
        if (threadIdx.x == 0) {
            half rmse1_sum = 0;
            half rmse2_sum = 0;
            for (int i = 0; i < embedding_dim; i++) {
                rmse1_sum += shared_input1[i] * shared_input1[i];
                rmse2_sum += shared_input2[i] * shared_input2[i];
            }
            shared_rmse1 = sqrt(rmse1_sum / embedding_dim);
            shared_rmse2 = sqrt(rmse2_sum / embedding_dim);
        }

        // Synchronize threads to ensure all shared memory values are available
        __syncthreads();

        // Normalize inputs and calculate cosine similarity
        half dot_product = 0;
        for (int i = 0; i < embedding_dim; i++) {
            dot_product += (shared_input1[i] / shared_rmse1) * (shared_input2[i] / shared_rmse2);
        }

        // Calculate cosine embedding loss
        half margin = 1.0;
        half target = labels[idx] == 1 ? 1.0 : -1.0;
        loss[idx] = max(half(0), margin - target * dot_product);
    }
}

// Function to calculate cosine embedding loss with RMSE normalization using Cutlass
void cosine_embedding_loss_rms_cutlass(const float *input1_data, const float *input2_data, const int *labels_data,
                                     float *loss_data, int batch_size, int embedding_dim) {
    // Define Cutlass types and layouts
    using Element = half;
    using LayoutA = cutlass::layout::TensorNHWC;
    using LayoutB = cutlass::layout::TensorNHWC;
    using LayoutC = cutlass::layout::TensorN;

    // Define Cutlass Gemm operation parameters
    cutlass::gemm::GemmCoord problem_size{batch_size, 1, embedding_dim};
    cutlass::gemm::GemmShape problem_shape{batch_size, 1, embedding_dim};
    cutlass::gemm::GemmEpilogueOp epilogue = cutlass::gemm::GemmEpilogueOp::kLinearCombination; // Use linear combination

    // Define Cutlass Gemm operation instance
    cutlass::gemm::Gemm<Element, Element, Element, LayoutA, LayoutB, LayoutC, cutlass::arch::Sm75,
                      cutlass::gemm::GemmMode::kGemm, epilogue> gemm_instance;

    // Define Cutlass threadblock and tile sizes
    int threadblock_size = 128;
    int tile_size = 16; // Adjust tile size as needed for optimal performance

    // Allocate device memory for input tensors and loss output
    Element *d_input1 = allocate_device_memory<Element>(batch_size * embedding_dim);
    Element *d_input2 = allocate_device_memory<Element>(batch_size * embedding_dim);
    int *d_labels = allocate_device_memory<int>(batch_size);
    Element *d_loss = allocate_device_memory<Element>(batch_size);

    // Copy input data to device memory
    copy_to_device(reinterpret_cast<const Element*>(input1_data), d_input1, batch_size * embedding_dim);
    copy_to_device(reinterpret_cast<const Element*>(input2_data), d_input2, batch_size * embedding_dim);
    copy_to_device(labels_data, d_labels, batch_size);

    // Create Cutlass tensor views for input tensors
    cutlass::util::HostTensor<Element, LayoutA> h_input1(problem_shape);
    cutlass::util::HostTensor<Element, LayoutB> h_input2(problem_shape);
    cutlass::util::HostTensor<Element, LayoutC> h_loss(batch_size);

    // Create Cutlass tensor views for device memory
    cutlass::util::TensorView<Element, LayoutA> d_input1_view(d_input1, problem_size);
    cutlass::util::TensorView<Element, LayoutB> d_input2_view(d_input2, problem_size);
    cutlass::util::TensorView<Element, LayoutC> d_loss_view(d_loss, batch_size);

    // Define Cutlass tile parameters
    cutlass::gemm::GemmCoord tile_extent{tile_size, tile_size, tile_size};

    // Define Cutlass threadblock parameters
    cutlass::gemm::GemmCoord threadblock_extent{tile_size, tile_size, 1};

    // Configure Cutlass Gemm operation
    gemm_instance.configure(problem_size, threadblock_extent, tile_extent);

    // Launch Cutlass Gemm operation
    cutlass::gemm::GemmLaunch(gemm_instance, h_input1, d_input1_view, h_input2, d_input2_view, 
                             h_loss, d_loss_view, threadblock_size);

    // Launch CUDA kernel to calculate cosine embedding loss
    cosine_embedding_loss_kernel<Element><<<batch_size / threadblock_size, threadblock_size>>>(d_loss, d_labels, d_loss, batch_size);

    // Copy loss output back to host memory
    copy_from_device(d_loss, reinterpret_cast<Element*>(loss_data), batch_size);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_labels);
    cudaFree(d_loss);
}

extern "C" {

void torch_cosine_embedding_rms_loss_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensors
    const float *input1 = va_arg(args, const float *);
    int input1_dim0 = va_arg(args, int);
    int input1_dim1 = va_arg(args, int);

    const float *input2 = va_arg(args, const float *);
    int input2_dim0 = va_arg(args, int);
    int input2_dim1 = va_arg(args, int);

    // Extract labels tensor
    const int *labels = va_arg(args, const int *);
    int labels_dim0 = va_arg(args, int);

    // Extract output tensor
    float *output = va_arg(args, float *);

    va_end(args);

    int batch_size = input1_dim0;
    int embedding_dim = input1_dim1;

    // Allocate device memory
    half *d_input1 = allocate_device_memory<half>(batch_size * embedding_dim);
    half *d_input2 = allocate_device_memory<half>(batch_size * embedding_dim);
    int *d_labels = allocate_device_memory<int>(batch_size);
    half *d_loss = allocate_device_memory<half>(batch_size);

    // Copy input data to device
    copy_to_device(reinterpret_cast<const half*>(input1), d_input1, batch_size * embedding_dim);
    copy_to_device(reinterpret_cast<const half*>(input2), d_input2, batch_size * embedding_dim);
    copy_to_device(labels, d_labels, batch_size);

    // Calculate cosine embedding loss with RMSE normalization
    cosine_embedding_loss_rms_cutlass(reinterpret_cast<const float*>(d_input1), reinterpret_cast<const float*>(d_input2),
                                      d_labels, reinterpret_cast<float*>(d_loss), batch_size, embedding_dim);

    // Copy result back to host
    copy_from_device(d_loss, reinterpret_cast<half*>(output), batch_size);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_labels);
    cudaFree(d_loss);
}

}  // extern "C"
