
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvFFT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvFFT, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.conv(x)
        x = torch.fft.fft2(x)  # Apply FFT on the spatial dimensions
        return x

def torch_conv_fft_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Applies convolution followed by FFT transformation.
    """
    torch.manual_seed(42)  # Seed for reproducibility

    # Convert to bfloat16 for memory efficiency
    input_tensor = input_tensor.to(torch.bfloat16)
    weight = weight.to(torch.bfloat16)
    bias = bias.to(torch.bfloat16)

    # Perform convolution
    output = F.conv2d(input_tensor, weight, bias=bias, stride=1, padding=1)

    # Apply FFT
    output = torch.fft.fft2(output)

    # Convert back to float32
    return output.to(torch.float32)


function_signature = {
    "name": "torch_conv_fft_function",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((3, 3, 3, 3), torch.float32),
        ((3,), torch.float32),
    ],
    "outputs": [
        ((1, 3, 224, 224), torch.complex64),
    ]
}
