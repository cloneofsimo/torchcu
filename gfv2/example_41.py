
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple

def conv3d_softplus_dilated(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, dilation: int, kernel_size: int) -> torch.Tensor:
    """
    Performs a 3D transposed convolution, followed by a softplus activation, and then a morphological dilation. 
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)
    output = F.conv_transpose3d(input_bf16, weight_bf16, bias_bf16, stride=stride, padding=padding, dilation=dilation)
    output = F.softplus(output)  # Softplus activation
    output = F.morphological_dilate(output, kernel_size=_triple(kernel_size))  # Morphological dilation
    return output.to(torch.float32)

function_signature = {
    "name": "conv3d_softplus_dilated",
    "inputs": [
        ((16, 16, 16, 3, 3), torch.float32),
        ((3, 3, 3, 3, 3), torch.float32),
        ((3,), torch.float32),
    ],
    "outputs": [
        ((16, 16, 16, 3, 3), torch.float32),
    ]
}
