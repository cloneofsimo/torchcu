
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel.h>
#include <cutlass/conv/conv2d.h>
#include <cutlass/epilogue/threadblock/linear_combine.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/transform/threadblock/copy.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/tensor_view.h>
#include <cutlass/util/reference/conv2d.h>
#include <cutlass/util/reference/gemm.h>
#include <iostream>

// Define the type for the input, output, and weights
using Element = float;
using Layout = cutlass::layout::TensorNHWC;

// Define the dimensions of the input tensor
constexpr int batch_size = 32;
constexpr int in_channels = 128;
constexpr int in_height = 14;
constexpr int in_width = 14;

// Define the dimensions of the output tensor
constexpr int out_channels = 128;
constexpr int out_height = 14;
constexpr int out_width = 14;

// Define the reduction factor for the SE module
constexpr int reduction_factor = 16;

// Define the dimensions of the intermediate tensors
constexpr int squeeze_size = out_channels / reduction_factor;

// Define the types for the convolutions
using Conv2D = cutlass::conv::kernel::Conv2D<cutlass::gemm::Gemm<
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    Element, Element, Element,
    cutlass::epilogue::threadblock::LinearCombine<
        cutlass::epilogue::threadblock::LinearCombineMode::kParallel,
        Element, Element, Element,
        cutlass::layout::RowMajor,
        cutlass::arch::Sm75,
        cutlass::epilogue::threadblock::LinearCombineTileIterator::kStrided,
        Element, Element>,
    cutlass::gemm::GemmThreadblockSwizzle::kStrided,
    cutlass::gemm::GemmThreadblockSwizzle::kStrided>>;

using Conv2DDesc = Conv2D::Params;
using Conv2DPlan = Conv2D::Plan;
using Conv2DProblem = cutlass::conv::kernel::Conv2DProblem;
using Conv2DArguments = Conv2D::Arguments;

// Define the types for the GEMMs
using Gemm = cutlass::gemm::Gemm<
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    Element, Element, Element,
    cutlass::epilogue::threadblock::LinearCombine<
        cutlass::epilogue::threadblock::LinearCombineMode::kParallel,
        Element, Element, Element,
        cutlass::layout::RowMajor,
        cutlass::arch::Sm75,
        cutlass::epilogue::threadblock::LinearCombineTileIterator::kStrided,
        Element, Element>,
    cutlass::gemm::GemmThreadblockSwizzle::kStrided,
    cutlass::gemm::GemmThreadblockSwizzle::kStrided>;

using GemmDesc = Gemm::Params;
using GemmPlan = Gemm::Plan;
using GemmProblem = cutlass::gemm::GemmProblem;
using GemmArguments = Gemm::Arguments;

// Define the types for the copy operations
using Copy = cutlass::transform::threadblock::Copy<
    cutlass::layout::TensorNHWC,
    cutlass::layout::RowMajor,
    Element,
    cutlass::arch::Sm75,
    cutlass::transform::threadblock::CopyTileIterator::kStrided,
    cutlass::transform::threadblock::CopyTileIterator::kStrided>;

using CopyDesc = Copy::Params;
using CopyPlan = Copy::Plan;
using CopyProblem = cutlass::transform::threadblock::CopyProblem;
using CopyArguments = Copy::Arguments;

// Define the types for the tensor views
using TensorView = cutlass::util::TensorView<Element, Layout>;

// Define the types for the host tensors
using HostTensor = cutlass::util::HostTensor<Element, Layout>;

__global__ void launch_conv(Element* input, Element* weight, Element* output, int batch_size, int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int stride, int padding, int dilation)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = bx * blockDim.x + tx;
    const int y = by * blockDim.y + ty;

    if (x < out_width && y < out_height)
    {
        // Calculate the input region
        const int in_x = x * stride - padding;
        const int in_y = y * stride - padding;

        // Loop over the output channels
        for (int oc = 0; oc < out_channels; oc++)
        {
            // Loop over the input channels
            float sum = 0.0f;
            for (int ic = 0; ic < in_channels; ic++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        // Calculate the input indices
                        const int in_x_k = in_x + k * dilation;
                        const int in_y_l = in_y + l * dilation;

                        // Check if the input indices are within the bounds
                        if (in_x_k >= 0 && in_x_k < in_width && in_y_l >= 0 && in_y_l < in_height)
                        {
                            // Compute the sum of the products
                            sum += input[((by * blockDim.y + ty) * in_width + in_x_k) * in_channels + ic] *
                                weight[(oc * 3 * 3 + k * 3 + l) * in_channels + ic];
                        }
                    }
                }
            }
            output[(y * out_width + x) * out_channels + oc] = sum;
        }
    }
}

__global__ void launch_squeeze(Element* input, Element* output, int batch_size, int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int stride, int padding, int dilation)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = bx * blockDim.x + tx;
    const int y = by * blockDim.y + ty;

    if (x < out_width && y < out_height)
    {
        // Calculate the input region
        const int in_x = x * stride - padding;
        const int in_y = y * stride - padding;

        // Loop over the output channels
        for (int oc = 0; oc < out_channels; oc++)
        {
            // Loop over the input channels
            float sum = 0.0f;
            for (int ic = 0; ic < in_channels; ic++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        // Calculate the input indices
                        const int in_x_k = in_x + k * dilation;
                        const int in_y_l = in_y + l * dilation;

                        // Check if the input indices are within the bounds
                        if (in_x_k >= 0 && in_x_k < in_width && in_y_l >= 0 && in_y_l < in_height)
                        {
                            // Compute the sum of the products
                            sum += input[((by * blockDim.y + ty) * in_width + in_x_k) * in_channels + ic] *
                                weight[(oc * 3 * 3 + k * 3 + l) * in_channels + ic];
                        }
                    }
                }
            }
            output[(y * out_width + x) * out_channels + oc] = sum;
        }
    }
}

__global__ void launch_excitation(Element* input, Element* weight, Element* output, int batch_size, int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int stride, int padding, int dilation)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = bx * blockDim.x + tx;
    const int y = by * blockDim.y + ty;

    if (x < out_width && y < out_height)
    {
        // Calculate the input region
        const int in_x = x * stride - padding;
        const int in_y = y * stride - padding;

        // Loop over the output channels
        for (int oc = 0; oc < out_channels; oc++)
        {
            // Loop over the input channels
            float sum = 0.0f;
            for (int ic = 0; ic < in_channels; ic++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        // Calculate the input indices
                        const int in_x_k = in_x + k * dilation;
                        const int in_y_l = in_y + l * dilation;

                        // Check if the input indices are within the bounds
                        if (in_x_k >= 0 && in_x_k < in_width && in_y_l >= 0 && in_y_l < in_height)
                        {
                            // Compute the sum of the products
                            sum += input[((by * blockDim.y + ty) * in_width + in_x_k) * in_channels + ic] *
                                weight[(oc * 3 * 3 + k * 3 + l) * in_channels + ic];
                        }
                    }
                }
            }
            output[(y * out_width + x) * out_channels + oc] = sum;
        }
    }
}


__global__ void launch_fc1(Element* input, Element* weight, Element* output, int batch_size, int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int stride, int padding, int dilation)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = bx * blockDim.x + tx;
    const int y = by * blockDim.y + ty;

    if (x < out_width && y < out_height)
    {
        // Calculate the input region
        const int in_x = x * stride - padding;
        const int in_y = y * stride - padding;

        // Loop over the output channels
        for (int oc = 0; oc < out_channels; oc++)
        {
            // Loop over the input channels
            float sum = 0.0f;
            for (int ic = 0; ic < in_channels; ic++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        // Calculate the input indices
                        const int in_x_k = in_x + k * dilation;
                        const int in_y_l = in_y + l * dilation;

                        // Check if the input indices are within the bounds
                        if (in_x_k >= 0 && in_x_k < in_width && in_y_l >= 0 && in_y_l < in_height)
                        {
                            // Compute the sum of the products
                            sum += input[((by * blockDim.y + ty) * in_width + in_x_k) * in_channels + ic] *
                                weight[(oc * 3 * 3 + k * 3 + l) * in_channels + ic];
                        }
                    }
                }
            }
            output[(y * out_width + x) * out_channels + oc] = sum;
        }
    }
}

__global__ void launch_fc2(Element* input, Element* weight, Element* output, int batch_size, int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int stride, int padding, int dilation)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = bx * blockDim.x + tx;
    const int y = by * blockDim.y + ty;

    if (x < out_width && y < out_height)
    {
        // Calculate the input region
        const int in_x = x * stride - padding;
        const int in_y = y * stride - padding;

        // Loop over the output channels
        for (int oc = 0; oc < out_channels; oc++)
        {
            // Loop over the input channels
            float sum = 0.0f;
            for (int ic = 0; ic < in_channels; ic++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        // Calculate the input indices
                        const int in_x_k = in_x + k * dilation;
                        const int in_y_l = in_y + l * dilation;

                        // Check if the input indices are within the bounds
                        if (in_x_k >= 0 && in_x_k < in_width && in_y_l >= 0 && in_y_l < in_height)
                        {
                            // Compute the sum of the products
                            sum += input[((by * blockDim.y + ty) * in_width + in_x_k) * in_channels + ic] *
                                weight[(oc * 3 * 3 + k * 3 + l) * in_channels + ic];
                        }
                    }
                }
            }
            output[(y * out_width + x) * out_channels + oc] = sum;
        }
    }
}

__global__ void launch_sigmoid(Element* input, Element* output, int batch_size, int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int stride, int padding, int dilation)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = bx * blockDim.x + tx;
    const int y = by * blockDim.y + ty;

    if (x < out_width && y < out_height)
    {
        // Calculate the input region
        const int in_x = x * stride - padding;
        const int in_y = y * stride - padding;

        // Loop over the output channels
        for (int oc = 0; oc < out_channels; oc++)
        {
            // Loop over the input channels
            float sum = 0.0f;
            for (int ic = 0; ic < in_channels; ic++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        // Calculate the input indices
                        const int in_x_k = in_x + k * dilation;
                        const int in_y_l = in_y + l * dilation;

                        // Check if the input indices are within the bounds
                        if (in_x_k >= 0 && in_x_k < in_width && in_y_l >= 0 && in_y_l < in_height)
                        {
                            // Compute the sum of the products
                            sum += input[((by * blockDim.y + ty) * in_width + in_x_k) * in_channels + ic] *
                                weight[(oc * 3 * 3 + k * 3 + l) * in_channels + ic];
                        }
                    }
                }
            }
            output[(y * out_width + x) * out_channels + oc] = sum;
        }
    }
}

extern "C" {
    
void torch_se_module(const float *input, const int *input_shape, float *output) {
    // Allocate device memory
    Element *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * in_channels * in_height * in_width * sizeof(Element));
    cudaMalloc(&d_output, batch_size * in_channels * in_height * in_width * sizeof(Element));

    // Copy input data to device
    cudaMemcpy(d_input, input, batch_size * in_channels * in_height * in_width * sizeof(Element), cudaMemcpyHostToDevice);

    // Define the convolution problem
    Conv2DProblem problem{
        {in_height, in_width},
        {3, 3},
        {1, 1},
        {1, 1},
        {in_channels, out_channels}
    };

    // Define the convolution plan
    Conv2DPlan plan{problem, Conv2DDesc{cutlass::arch::Sm75}};

    // Allocate device memory for the weight tensor
    Element *d_weight;
    cudaMalloc(&d_weight, in_channels * out_channels * 9 * sizeof(Element));

    // Create a host tensor for the weight
    HostTensor weight_tensor{in_channels * out_channels * 9};

    // Initialize the weight tensor with random values
    for (int i = 0; i < in_channels * out_channels * 9; i++) {
        weight_tensor.at(i) = static_cast<Element>(rand() / (RAND_MAX + 1.0f));
    }

    // Copy the weight tensor to the device
    cudaMemcpy(d_weight, weight_tensor.data(), in_channels * out_channels * 9 * sizeof(Element), cudaMemcpyHostToDevice);

    // Define the GEMM problem for the squeeze operation
    GemmProblem squeeze_problem{
        {in_channels * in_height * in_width, squeeze_size},
        {squeeze_size, in_channels * in_height * in_width},
        {batch_size, 1},
        {1, 1}
    };

    // Define the GEMM plan for the squeeze operation
    GemmPlan squeeze_plan{squeeze_problem, GemmDesc{cutlass::arch::Sm75}};

    // Allocate device memory for the squeeze weights
    Element *d_squeeze_weight;
    cudaMalloc(&d_squeeze_weight, squeeze_size * in_channels * in_height * in_width * sizeof(Element));

    // Create a host tensor for the squeeze weights
    HostTensor squeeze_weight_tensor{squeeze_size * in_channels * in_height * in_width};

    // Initialize the squeeze weights with random values
    for (int i = 0; i < squeeze_size * in_channels * in_height * in_width; i++) {
        squeeze_weight_tensor.at(i) = static_cast<Element>(rand() / (RAND_MAX + 1.0f));
    }

    // Copy the squeeze weights to the device
    cudaMemcpy(d_squeeze_weight, squeeze_weight_tensor.data(), squeeze_size * in_channels * in_height * in_width * sizeof(Element), cudaMemcpyHostToDevice);

    // Define the GEMM problem for the excitation operation
    GemmProblem excitation_problem{
        {squeeze_size, in_channels * in_height * in_width},
        {in_channels * in_height * in_width, batch_size},
        {1, 1},
        {1, 1}
    };

    // Define the GEMM plan for the excitation operation
    GemmPlan excitation_plan{excitation_problem, GemmDesc{cutlass::arch::Sm75}};

    // Allocate device memory for the excitation weights
    Element *d_excitation_weight;
    cudaMalloc(&d_excitation_weight, in_channels * in_height * in_width * batch_size * sizeof(Element));

    // Create a host tensor for the excitation weights
    HostTensor excitation_weight_tensor{in_channels * in_height * in_width * batch_size};

    // Initialize the excitation weights with random values
    for (int i = 0; i < in_channels * in_height * in_width * batch_size; i++) {
        excitation_weight_tensor.at(i) = static_cast<Element>(rand() / (RAND_MAX + 1.0f));
    }

    // Copy the excitation weights to the device
    cudaMemcpy(d_excitation_weight, excitation_weight_tensor.data(), in_channels * in_height * in_width * batch_size * sizeof(Element), cudaMemcpyHostToDevice);

    // Define the convolution arguments
    Conv2DArguments arguments{
        d_input,
        d_weight,
        d_output
    };

    // Launch the convolution kernel
    plan.launch(arguments);

    // Allocate device memory for the squeeze output
    Element *d_squeeze_output;
    cudaMalloc(&d_squeeze_output, batch_size * squeeze_size * sizeof(Element));

    // Define the GEMM arguments for the squeeze operation
    GemmArguments squeeze_arguments{
        d_input,
        d_squeeze_weight,
        d_squeeze_output
    };

    // Launch the GEMM kernel for the squeeze operation
    squeeze_plan.launch(squeeze_arguments);

    // Allocate device memory for the relu output
    Element *d_relu_output;
    cudaMalloc(&d_relu_output, batch_size * squeeze_size * sizeof(Element));

    // Launch the ReLU kernel
    launch_relu<<<(batch_size * squeeze_size + 128 - 1) / 128, 128>>>(d_squeeze_output, d_relu_output, batch_size * squeeze_size);

    // Allocate device memory for the fc2 output
    Element *d_fc2_output;
    cudaMalloc(&d_fc2_output, batch_size * out_channels * sizeof(Element));

    // Define the GEMM arguments for the fc2 operation
    GemmArguments fc2_arguments{
        d_relu_output,
        d_excitation_weight,
        d_fc2_output
    };

    // Launch the GEMM kernel for the fc2 operation
    excitation_plan.launch(fc2_arguments);

    // Allocate device memory for the sigmoid output
    Element *d_sigmoid_output;
    cudaMalloc(&d_sigmoid_output, batch_size * out_channels * sizeof(Element));

    // Launch the sigmoid kernel
    launch_sigmoid<<<(batch_size * out_channels + 128 - 1) / 128, 128>>>(d_fc2_output, d_sigmoid_output, batch_size * out_channels);

    // Allocate device memory for the excitation output
    Element *d_excitation_output;
    cudaMalloc(&d_excitation_output, batch_size * out_channels * in_height * in_width * sizeof(Element));

    // Launch the excitation kernel
    launch_excitation<<<(batch_size * in_channels * in_height * in_width + 128 - 1) / 128, 128>>>(d_sigmoid_output, d_weight, d_excitation_output, batch_size, out_channels, in_height, in_width, out_channels, in_height, in_width, 1, 0, 1);

    // Multiply the excitation output with the original input
    launch_mul<<<(batch_size * in_channels * in_height * in_width + 128 - 1) / 128, 128>>>(d_excitation_output, d_input, d_output, batch_size * in_channels * in_height * in_width);

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * in_channels * in_height * in_width * sizeof(Element), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_squeeze_weight);
    cudaFree(d_squeeze_output);
    cudaFree(d_relu_output);
    cudaFree(d_excitation_weight);
    cudaFree(d_fc2_output);
    cudaFree(d_sigmoid_output);
    cudaFree(d_excitation_output);
}

}  // extern "C"

__global__ void launch_relu(Element* input, Element* output, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

__global__ void launch_mul(Element* input, Element* output, Element* result, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = input[i] * output[i];
    }
}
