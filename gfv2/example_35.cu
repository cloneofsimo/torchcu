
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

#include "cutlass.h"

extern "C" {

void exponential_nll_loss_fp16(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract target tensor
    const int* target_tensor = va_arg(args, const int*);
    int target_tensor_dim0 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    half* output = va_arg(args, half*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;

    // Allocate device memory
    float* d_input;
    int* d_target;
    half* d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_target, batch_size * sizeof(int));
    cudaMalloc(&d_output, sizeof(half));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_tensor, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // Define CUTLASS configuration
    cutlass::gemm::GemmCoord problem_size = {input_dim, batch_size, 1}; // N, M, K
    cutlass::layout::TensorNHWC  input_layout = cutlass::layout::TensorNHWC(input_dim); // Input tensor layout
    cutlass::layout::TensorNHWC  output_layout = cutlass::layout::TensorNHWC(1); // Output tensor layout
    cutlass::epilogue::LinearCombinationParams epilogue = cutlass::epilogue::LinearCombinationParams();
    cutlass::epilogue::ThreadblockOutputOp threadblock_op = cutlass::epilogue::ThreadblockOutputOp::kOutputOpNone;
    cutlass::epilogue::EpilogueOutputOp output_op = cutlass::epilogue::EpilogueOutputOp::kOutputOpNone;
    cutlass::epilogue::TileOutputOp tile_op = cutlass::epilogue::TileOutputOp::kOutputOpNone;
    cutlass::arch::SmArch_t arch = cutlass::arch::SmArch_t::sm_80; // Specify the SM architecture 

    // Instantiate Cutlass GEMM and NLL Loss operators
    using Gemm = cutlass::gemm::Gemm<cutlass::gemm::GemmShape<16, 16, 8>, 
                                     cutlass::layout::RowMajor, 
                                     cutlass::layout::RowMajor, 
                                     cutlass::layout::RowMajor, 
                                     cutlass::epilogue::LinearCombinationParams(),
                                     cutlass::arch::SmArch_t::sm_80, 
                                     cutlass::DataType::f16, 
                                     cutlass::DataType::f16, 
                                     cutlass::DataType::f32, 
                                     cutlass::DataType::f16>;

    // Instantiate a GEMM operation
    Gemm gemm_op; 
    gemm_op.set_problem_size(problem_size); 
    gemm_op.set_input_layout(input_layout); 
    gemm_op.set_output_layout(output_layout); 

    // NLL Loss calculation
    cutlass::nll::NllLossParams params{output_layout};
    cutlass::nll::NllLoss<cutlass::nll::NllLossParams, cutlass::DataType::f32> nll_loss_op(params);
    
    // Allocate Cutlass workspace
    void* workspace = nullptr;
    size_t workspace_size = gemm_op.get_workspace_size() + nll_loss_op.get_workspace_size();
    cudaMalloc(&workspace, workspace_size);

    // Convert input tensor to half precision
    half* d_input_fp16;
    cudaMalloc(&d_input_fp16, batch_size * input_dim * sizeof(half));
    cudaMemcpy(d_input_fp16, d_input, batch_size * input_dim * sizeof(half), cudaMemcpyDeviceToDevice);

    // Launch Cutlass GEMM operation 
    gemm_op.execute(d_input_fp16, workspace, workspace_size, d_input_fp16);
    
    // Launch NLL Loss operation
    nll_loss_op.execute(d_input_fp16, d_target, workspace, workspace_size, d_output, batch_size);

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_output);
    cudaFree(workspace);
    cudaFree(d_input_fp16);
}

} // extern "C"
