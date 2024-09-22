
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end
#include <cutlass/cutlass.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weight1 tensor
    const float* weight1 = va_arg(args, const float*);
    int weight1_dim0 = va_arg(args, int);
    int weight1_dim1 = va_arg(args, int);

    // Extract weight2 tensor
    const float* weight2 = va_arg(args, const float*);
    int weight2_dim0 = va_arg(args, int);
    int weight2_dim1 = va_arg(args, int);

    // Extract gamma (scalar)
    float gamma = va_arg(args, double);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight1_dim0 + weight2_dim0;

    // Allocate device memory
    float *d_input, *d_weight1, *d_weight2, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight1, weight1_dim0 * input_dim * sizeof(float));
    cudaMalloc(&d_weight2, weight2_dim0 * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight1, weight1, weight1_dim0 * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight2, weight2, weight2_dim0 * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Fading-out
    float fading_factor = exp(-gamma * input_dim);
    cudaMemset(d_output, 0, batch_size * output_dim * sizeof(float)); // Zero out the output tensor
    
    cutlass::gemm::GemmConfig config;
    config.epilogue = cutlass::epilogue::Identity;
    config.layout_a = cutlass::layout::RowMajor;
    config.layout_b = cutlass::layout::ColumnMajor;

    // Matmul 1
    cutlass::gemm::GemmPlan<
        cutlass::gemm::GemmShape<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>,
        cutlass::half_t, 
        cutlass::half_t, 
        cutlass::float_t,
        cutlass::arch::Sm80
    > plan(config);

    cutlass::gemm::Gemm<
        cutlass::gemm::GemmShape<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>,
        cutlass::half_t, 
        cutlass::half_t, 
        cutlass::float_t,
        cutlass::arch::Sm80
    > gemm(plan);
    
    // Launch Cutlass GEMM kernels
    gemm(
        batch_size, input_dim, weight1_dim0, fading_factor,  // M, N, K, alpha (fading factor)
        d_input, d_weight1, d_output,                        // A, B, C
        d_output, d_output,                                  // D, E (for accumulation)
        cudaStreamDefault
    );

    // Matmul 2
    gemm.set_C(d_output + weight1_dim0);  // Set the starting address for C
    gemm(
        batch_size, input_dim, weight2_dim0, fading_factor,  // M, N, K, alpha (fading factor)
        d_input, d_weight2, d_output + weight1_dim0,     // A, B, C
        d_output + weight1_dim0, d_output + weight1_dim0, // D, E (for accumulation)
        cudaStreamDefault
    );

    // ReLU activation
    cudaMemset(d_output, 0, batch_size * output_dim * sizeof(float)); 
    cutlass::activation::Relu<cutlass::float_t, cutlass::arch::Sm80> relu;
    relu.run(d_output, d_output, batch_size * output_dim);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight1);
    cudaFree(d_weight2);
    cudaFree(d_output);
}

}  // extern "C"
