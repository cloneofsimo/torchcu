
import torch
import torch.nn.functional as F
from torch.fft import fft, ifft
from cutlass import *

def torch_conv1d_fft_sqrt_coord_conv(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int) -> torch.Tensor:
    """
    Performs a 1D convolution using FFT, applies square root, and then applies a coordinate convolution.
    """
    # Convert to bfloat16
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    # 1. Convolution using FFT
    output = F.conv1d(input_bf16, weight_bf16, bias_bf16, stride=stride, padding=padding, dilation=dilation)

    # 2. Square root
    output = torch.sqrt(output)

    # 3. Coordinate Convolution
    batch_size, channels, seq_len = output.shape
    coord_weight = torch.ones(1, channels, kernel_size, dtype=torch.bfloat16)
    coord_bias = torch.zeros(1, channels, dtype=torch.bfloat16)
    output = F.conv1d(output, coord_weight, coord_bias, stride=1, padding=kernel_size // 2)

    return output.to(torch.float32)

function_signature = {
    "name": "torch_conv1d_fft_sqrt_coord_conv",
    "inputs": [
        ((16, 32, 128), torch.float32),
        ((32, 32, 5), torch.float32),
        ((32,), torch.float32),
        (5,),
        (2,),
        (1,),
        (1,),
    ],
    "outputs": [
        ((16, 32, 128), torch.float32),
    ]
}
