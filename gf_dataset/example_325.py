
import torch
import torch.nn as nn
import torch.nn.functional as F
from cutlass import *
import cutlass

class SeparableConv2d_bf16(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d_bf16, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def torch_separable_conv2d_bf16_function(input_tensor: torch.Tensor, depthwise_weight: torch.Tensor, pointwise_weight: torch.Tensor) -> torch.Tensor:
    """
    Perform a separable convolution operation with bfloat16 precision
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    depthwise_weight_bf16 = depthwise_weight.to(torch.bfloat16)
    pointwise_weight_bf16 = pointwise_weight.to(torch.bfloat16)

    # Use nn.functional.conv2d for depthwise convolution
    output = F.conv2d(input_bf16, depthwise_weight_bf16, groups=input_bf16.shape[1], stride=1, padding=1)

    # Use nn.functional.conv2d for pointwise convolution
    output = F.conv2d(output, pointwise_weight_bf16, stride=1, padding=0)

    return output.to(torch.float32)

function_signature = {
    "name": "torch_separable_conv2d_bf16_function",
    "inputs": [
        ((1, 3, 16, 16), torch.float32),
        ((3, 3, 1, 1), torch.float32),
        ((1, 12, 1, 1), torch.float32),
    ],
    "outputs": [
        ((1, 12, 14, 14), torch.float32)
    ]
}
