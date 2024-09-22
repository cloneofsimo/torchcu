
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <device_launch_parameters.h>

// CUDA kernel for tanh activation
__global__ void tanh_kernel(const float* input, float* output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = tanhf(input[i]);
  }
}

// CUDA kernel for comparing with threshold and calculating ZCR
__global__ void compare_zcr_kernel(const float* input, const float threshold, 
                                     float* output_eq, float* zcr, int batch_size, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    int batch_idx = i / size;
    int element_idx = i % size;

    // Compare with threshold
    output_eq[i] = (input[i] == threshold) ? 1.0f : 0.0f;

    // Calculate ZCR (only for elements after the first)
    if (element_idx > 0) {
      int prev_idx = batch_idx * size + element_idx - 1;
      zcr[batch_idx] += (input[i] * input[prev_idx] < 0) ? 1.0f : 0.0f;
    }
  }
}

// CUDA kernel for dividing ZCR by the size (to get the average)
__global__ void zcr_average_kernel(float* zcr, int batch_size, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size) {
    zcr[i] /= (float)size;
  }
}

// This function is for Cutlass matrix multiplication
template <typename T>
void matmul_cutlass(T* input, T* weight, T* output, int m, int k, int n) {
  // Define types and sizes for Cutlass
  cutlass::MatrixLayout layout_m = cutlass::kRowMajor;
  cutlass::MatrixLayout layout_n = cutlass::kRowMajor;
  cutlass::MatrixLayout layout_k = cutlass::kRowMajor;
  cutlass::int32_t m_ = m;
  cutlass::int32_t k_ = k;
  cutlass::int32_t n_ = n;

  // Define the GEMM operation
  using ElementA = T;
  using ElementB = T;
  using ElementC = T;
  using ElementAccumulator = T;
  using EpilogueOp = cutlass::epilogue::Identity<ElementC, ElementAccumulator, ElementA>;
  using Gemm = cutlass::gemm::Gemm<
      cutlass::gemm::GemmShape<m_, k_, n_>,
      cutlass::gemm::GemmLayout<layout_m, layout_n, layout_k>,
      cutlass::gemm::GemmOp<ElementA, ElementB, ElementC, ElementAccumulator, EpilogueOp>
  >;

  // Instantiate the Gemm operation
  Gemm gemm;

  // Initialize the Gemm operation
  gemm.initialize();

  // Define the workspace
  int workspace_size = gemm.get_workspace_size();
  void* workspace = nullptr;
  if (workspace_size > 0) {
    workspace = malloc(workspace_size);
  }

  // Launch the GEMM operation
  gemm.execute(
      cutlass::TensorRef<ElementA, cutlass::layout::RowMajor>(input, m_, k_),
      cutlass::TensorRef<ElementB, cutlass::layout::RowMajor>(weight, k_, n_),
      cutlass::TensorRef<ElementC, cutlass::layout::RowMajor>(output, m_, n_),
      workspace
  );

  // Free the workspace
  if (workspace_size > 0) {
    free(workspace);
  }
}

extern "C" {
  
  void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract threshold
    float threshold = va_arg(args, float);

    // Extract output tensors
    float* output_tanh = va_arg(args, float*);
    float* zcr = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int size = input_tensor_dim1;

    // Allocate device memory
    float* d_input;
    cudaMalloc(&d_input, batch_size * size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch tanh kernel
    tanh_kernel<<<(size + 255) / 256, 256>>>(d_input, output_tanh, batch_size * size);

    // Launch compare_zcr kernel
    compare_zcr_kernel<<<(batch_size * size + 255) / 256, 256>>>(
        output_tanh, threshold, output_tanh, zcr, batch_size, size
    );

    // Launch zcr_average kernel
    zcr_average_kernel<<<(batch_size + 255) / 256, 256>>>(zcr, batch_size, size);

    // Copy result back to host
    cudaMemcpy(output_tanh, output_tanh, batch_size * size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(zcr, zcr, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
  }
}
