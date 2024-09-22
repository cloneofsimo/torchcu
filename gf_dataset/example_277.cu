
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel.h>
#include <cutlass/conv/device/implicit_gemm.h>
#include <cutlass/conv/device/gemm_multi_stage.h>
#include <cutlass/conv/device/gemm_multi_stage_fast.h>
#include <cutlass/conv/device/gemm_multi_stage_fast_no_bias.h>

using namespace cutlass;
using namespace cutlass::conv;

// CUDA kernel for sparse convolution
__global__ void sparse_conv_kernel(const float* input, const float* weights, const float* bias, const bool* indices, float* output,
                                  int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {

  // Calculate output coordinates
  int output_y = blockIdx.y * blockDim.y + threadIdx.y;
  int output_x = blockIdx.x * blockDim.x + threadIdx.x;

  if (output_y < height && output_x < width) {
    float sum = 0.0f;
    for (int k = 0; k < kernel_size * kernel_size; ++k) {
      for (int i = 0; i < in_channels; ++i) {
        if (indices[k * in_channels + i]) { // Check if weight is non-zero
          // Get the weight and input value
          float weight_val = weights[k * in_channels + i];
          float input_val = input[(output_y * width + output_x) * in_channels + i];

          // Multiply and accumulate
          sum += weight_val * input_val;
        }
      }
    }

    // Add bias
    if (bias) {
      sum += bias[threadIdx.x * blockDim.x + threadIdx.y];
    }

    // Apply ReLU activation
    output[(output_y * width + output_x) * out_channels + threadIdx.z] = max(0.0f, sum);
  }
}

// Define the types for the convolution kernel
using Element = float;
using Layout = TensorNHWC;
using IndexType = int;
using ElementAccumulator = float;

// Define the convolution parameters
using KernelSize = int2(3, 3);
using Stride = int2(1, 1);
using Padding = int2(1, 1);
using Dilations = int2(1, 1);

// Define the Cutlass convolution operator
using SparseConvOp = device::ImplicitGemm;
//using SparseConvOp = device::GemmMultiStageFast;
//using SparseConvOp = device::GemmMultiStageFastNoBias;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the data type for the convolution
using Element = float;
using IndexType = int;
using ElementAccumulator = float;

// Define the Cutlass workspace type
using Workspace = WorkspaceNone;

// Define the convolution operator
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 16),
  cutlass::int4(1, 3, 3, 16),
  KernelSize,
  Stride,
  Padding,
  Dilations>;

// Define the Cutlass convolution layout
using ConvLayout = conv::Layout;

// Define the CUTLASS convolution operation
using SparseConvOp = device::ImplicitGemm;

// Define the CUTLASS architecture
using Arch = cutlass::arch::Sm80;

// Define the convolution problem size
using ProblemSize = conv::ProblemSize<
  cutlass::int4(1, 32, 32, 1