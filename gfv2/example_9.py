
import torch
import torch.nn.functional as F
from cutlass import *

def conv3d_fp16_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Perform a 3D convolution with fp16 precision and bias addition.
    """
    input_fp16 = input_tensor.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)
    bias_fp16 = bias.to(torch.float16)
    output = F.conv3d(input_fp16, weight_fp16, bias_fp16, stride=1, padding=1)
    return output.to(torch.float32)

function_signature = {
    "name": "conv3d_fp16_function",
    "inputs": [
        ((1, 16, 10, 10, 10), torch.float32),
        ((16, 16, 3, 3, 3), torch.float32),
        ((16,), torch.float32)
    ],
    "outputs": [
        ((1, 16, 10, 10, 10), torch.float32)
    ]
}
