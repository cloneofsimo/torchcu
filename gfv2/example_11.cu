
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv2d_wgrad.h>
#include <cutlass/conv/kernel/default_conv2d_dgrad.h>
#include <cutlass/conv/threadblock/default_conv2d_threadblock.h>
#include <cutlass/conv/tile_iterator/default_conv2d_tile_iterator.h>
#include <cutlass/conv/warp/default_conv2d_warp.h>
#include <cutlass/conv/epilogue/default_conv2d_epilogue.h>
#include <cutlass/epilogue/threadblock/epilogue_threadblock_identity.h>
#include <cutlass/epilogue/threadblock/epilogue_threadblock_eltwise.h>
#include <cutlass/epilogue/threadblock/epilogue_threadblock_linear.h>
#include <cutlass/epilogue/threadblock/epilogue_threadblock_fast_linear.h>
#include <cutlass/epilogue/warp/epilogue_warp_identity.h>
#include <cutlass/epilogue/warp/epilogue_warp_eltwise.h>
#include <cutlass/epilogue/warp/epilogue_warp_linear.h>
#include <cutlass/epilogue/warp/epilogue_warp_fast_linear.h>
#include <cutlass/transform/threadblock/transform_threadblock_identity.h>
#include <cutlass/transform/threadblock/transform_threadblock_linear.h>
#include <cutlass/transform/threadblock/transform_threadblock_fast_linear.h>
#include <cutlass/transform/warp/transform_warp_identity.h>
#include <cutlass/transform/warp/transform_warp_linear.h>
#include <cutlass/transform/warp/transform_warp_fast_linear.h>
#include <cutlass/conv/tensor_op/tensor_op_identity.h>
#include <cutlass/conv/tensor_op/tensor_op_eltwise.h>
#include <cutlass/conv/tensor_op/tensor_op_linear.h>
#include <cutlass/conv/tensor_op/tensor_op_fast_linear.h>
#include <cutlass/layout/TensorNHWC.h>
#include <cutlass/layout/TensorNCHW.h>
#include <cutlass/layout/TensorNC.h>
#include <cutlass/layout/TensorN.h>
#include <cutlass/gemm/device/gemm.h>

#include <cmath>
#include <cstdlib>
#include <cassert>
#include <cstdio>

#include <algorithm>
#include <limits>
#include <vector>
#include <stdexcept>

#include <iostream>
#include <fstream>
#include <sstream>

#include <stdarg.h>  // Add this for va_list, va_start, va_end

#define CUDA_CHECK(x)                                                                \
  {                                                                               \
    cudaError_t error = (x);                                                      \
    if (error != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));   \
      exit(EXIT_FAILURE);                                                        \
    }                                                                               \
  }

#define CUTLASS_CHECK(x)                                                          \
  {                                                                               \
    cutlass::Status status = (x);                                                 \
    if (status != cutlass::Status::kSuccess) {                                   \
      fprintf(stderr, "CUTLASS error: %s:%d: %s\n", __FILE__, __LINE__, status.message().c_str()); \
      exit(EXIT_FAILURE);                                                        \
    }                                                                               \
  }

#define CUTLASS_CHECK_MSG(x, msg)                                                \
  {                                                                               \
    cutlass::Status status = (x);                                                 \
    if (status != cutlass::Status::kSuccess) {                                   \
      fprintf(stderr, "CUTLASS error: %s:%d: %s\n", __FILE__, __LINE__, status.message().c_str()); \
      fprintf(stderr, msg);                                                    \
      exit(EXIT_FAILURE);                                                        \
    }                                                                               \
  }


extern "C" {
  void torch_mel_spectrogram_cutlass(int num_args, ...) {
      va_list args;
      va_start(args, num_args);

      // Extract input tensor
      const float* input_tensor = va_arg(args, const float*);
      int input_tensor_dim0 = va_arg(args, int);
      int input_tensor_dim1 = va_arg(args, int);

      // Extract sample rate
      int sample_rate = va_arg(args, int);
      int n_fft = va_arg(args, int);
      int hop_length = va_arg(args, int);
      int win_length = va_arg(args, int);
      int n_mels = va_arg(args, int);
      float f_min = va_arg(args, double);
      float f_max = va_arg(args, double);
      bool center = va_arg(args, int);
      const char* pad_mode = va_arg(args, char*);
      float power = va_arg(args, double);

      // Extract output tensor (assuming it's preallocated)
      float* output = va_arg(args, float*);

      va_end(args);

      int batch_size = input_tensor_dim0;
      int input_dim = input_tensor_dim1;

      // Allocate device memory
      float *d_input, *d_output;
      CUDA_CHECK(cudaMalloc(&d_input, batch_size * input_dim * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_output, batch_size * n_mels * (input_dim/hop_length - 1) * sizeof(float)));

      // Copy input data to device
      CUDA_CHECK(cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));

      // --- Cutlass Mel Spectrogram Computation ---

      // Define the shape of the input and output tensors
      int input_channels = 1;
      int output_channels = n_mels;
      int filter_size = n_fft;

      // Kernel configuration for Cutlass
      cutlass::conv::kernel::DefaultConv2d<
          cutlass::layout::TensorNHWC,
          cutlass::layout::TensorNCHW,
          cutlass::layout::TensorNHWC,
          cutlass::float32_t,
          cutlass::float32_t,
          cutlass::float32_t,
          cutlass::arch::Sm80,
          cutlass::epilogue::threadblock::EpilogueThreadblockIdentity<cutlass::float32_t>,
          cutlass::transform::threadblock::TransformThreadblockIdentity<cutlass::float32_t>,
          cutlass::conv::tensor_op::TensorOpIdentity<cutlass::float32_t>,
          cutlass::conv::warp::WarpDefaultConv2d<cutlass::float32_t,
          cutlass::epilogue::warp::EpilogueWarpIdentity<cutlass::float32_t>>,
          cutlass::conv::tile_iterator::DefaultConv2dTileIterator<cutlass::float32_t>,
          cutlass::conv::threadblock::DefaultConv2dThreadblock<cutlass::float32_t,
          cutlass::arch::Sm80,
          cutlass::epilogue::threadblock::EpilogueThreadblockIdentity<cutlass::float32_t>,
          cutlass::transform::threadblock::TransformThreadblockIdentity<cutlass::float32_t>,
          cutlass::conv::tensor_op::TensorOpIdentity<cutlass::float32_t>>> conv_kernel;

      // Define the problem size and number of threads for Cutlass kernel
      cutlass::conv::gemm::GemmCoord problem_size{batch_size, output_channels, input_channels * filter_size};
      cutlass::gemm::GemmCoord threadblock_shape{128, 1, 1};

      // Create a Cutlass GEMM operator and configure it
      cutlass::gemm::device::Gemm<cutlass::float32_t, cutlass::float32_t, cutlass::float32_t,
      cutlass::arch::Sm80,
      cutlass::layout::TensorNCHW,
      cutlass::layout::TensorNCHW,
      cutlass::layout::TensorNCHW,
      cutlass::gemm::GemmShape<128, 128, 128>,
      cutlass::gemm::GemmShape<64, 16, 32>,
      cutlass::gemm::GemmShape<64, 1, 32>> gemm_op;

      // Define workspace size for Cutlass
      size_t workspace_size = gemm_op.get_workspace_size(problem_size, threadblock_shape);
      void *workspace = nullptr;
      CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

      // Define pointers for the Cutlass GEMM operation
      cutlass::float32_t* d_weights = reinterpret_cast<cutlass::float32_t*>(d_input);
      cutlass::float32_t* d_activations = reinterpret_cast<cutlass::float32_t*>(d_input);
      cutlass::float32_t* d_output_cutlass = reinterpret_cast<cutlass::float32_t*>(d_output);

      // Launch the Cutlass GEMM operation
      gemm_op.run(problem_size, threadblock_shape, d_weights, d_activations, d_output_cutlass, workspace);

      // Free workspace memory
      CUDA_CHECK(cudaFree(workspace));

      // --- End Cutlass Mel Spectrogram Computation ---

      // Copy result back to host
      CUDA_CHECK(cudaMemcpy(output, d_output, batch_size * n_mels * (input_dim/hop_length - 1) * sizeof(float), cudaMemcpyDeviceToHost));

      // Free device memory
      CUDA_CHECK(cudaFree(d_input));
      CUDA_CHECK(cudaFree(d_output));
  }
}  // extern "C"
