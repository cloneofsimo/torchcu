
import torch

def depthwise_separable_conv_int8(input_tensor: torch.Tensor, depthwise_weight: torch.Tensor, pointwise_weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a depthwise separable convolution with int8 quantization for efficiency.
    """

    # Quantize input and weights to int8
    input_int8 = input_tensor.to(torch.int8)
    depthwise_weight_int8 = depthwise_weight.to(torch.int8)
    pointwise_weight_int8 = pointwise_weight.to(torch.int8)

    # Perform depthwise convolution
    depthwise_output = torch.nn.functional.conv2d(input_int8, depthwise_weight_int8, groups=input_tensor.shape[1])

    # Perform pointwise convolution
    pointwise_output = torch.nn.functional.conv2d(depthwise_output, pointwise_weight_int8)

    # Dequantize output to fp32
    output = pointwise_output.to(torch.float32)
    return output

function_signature = {
    "name": "depthwise_separable_conv_int8",
    "inputs": [
        ((1, 16, 10, 10), torch.float32),
        ((16, 1, 3, 3), torch.float32),
        ((32, 16, 1, 1), torch.float32)
    ],
    "outputs": [
        ((1, 32, 10, 10), torch.float32)
    ]
}
