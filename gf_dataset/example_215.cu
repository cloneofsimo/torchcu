
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

#define CHECK(x) \
  do\
  {\
    cudaError_t err = (x);\
    if (err != cudaSuccess)\
    {\
      fprintf(stderr, "Error: %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
      exit(EXIT_FAILURE);\
    }\
  } while (0)

extern "C" {

void pixel_shuffle_torch(int num_args, ...) {
  va_list args;
  va_start(args, num_args);

  const float* input_tensor = va_arg(args, const float*);
  int batch_size = va_arg(args, int);
  int in_channels = va_arg(args, int);
  int in_height = va_arg(args, int);
  int in_width = va_arg(args, int);
  int upscale_factor = va_arg(args, int);

  float* output_tensor = va_arg(args, float*);

  va_end(args);

  int out_channels = in_channels / (upscale_factor * upscale_factor);
  int out_height = in_height * upscale_factor;
  int out_width = in_width * upscale_factor;

  // Allocate device memory
  float* d_input;
  CHECK(cudaMalloc(&d_input, batch_size * in_channels * in_height * in_width * sizeof(float)));
  float* d_output;
  CHECK(cudaMalloc(&d_output, batch_size * out_channels * out_height * out_width * sizeof(float)));

  // Copy input data to device
  CHECK(cudaMemcpy(d_input, input_tensor, batch_size * in_channels * in_height * in_width * sizeof(float), cudaMemcpyHostToDevice));

  // Perform pixel shuffle using Cutlass
  // (You'll need to adapt this based on your specific Cutlass configuration)
  // Note: This example assumes a simple 2x2 upscale factor
  cutlass::epilogue::threadblock::Identity<float, cutlass::layout::RowMajor> epilogue;
  cutlass::gemm::GemmConfig config = {
      in_channels / 4,   // M: Input channels / 4
      in_height * in_width, // N: Input height * input width
      4,                // K: 4 (for 2x2 upscale)
      2,                // Block size
      2,                // Block size
      cutlass::layout::RowMajor,
      cutlass::layout::RowMajor,
      cutlass::layout::RowMajor,
      cutlass::arch::Sm80,
      cutlass::epilogue::ThreadblockOp::Identity
  };
  cutlass::gemm::GemmPlan<float, float, float> plan;
  CHECK(cutlass::gemm::GemmPlan::make(config, epilogue, plan));

  cutlass::gemm::GemmArguments args;
  args.A = d_input;
  args.B = reinterpret_cast<float*>(d_input + (in_channels / 4) * in_height * in_width);
  args.C = d_output;
  args.ldA = in_channels / 4;
  args.ldB = 4;
  args.ldC = out_channels;

  cutlass::gemm::GemmOperation<float, float, float, cutlass::arch::Sm80>::launch(plan, args);

  // Copy result back to host
  CHECK(cudaMemcpy(output_tensor, d_output, batch_size * out_channels * out_height * out_width * sizeof(float), cudaMemcpyDeviceToHost));

  // Free device memory
  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_output));
}

}  // extern "C"
