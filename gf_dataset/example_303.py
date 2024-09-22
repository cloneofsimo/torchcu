
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def torch_grouped_conv1d_bf16_function(input_tensor: torch.Tensor, weight: torch.Tensor, groups: int, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Performs a grouped 1D convolution with bfloat16 precision and ReLU activation.
    """
    with autocast():
        output = F.conv1d(input_tensor.to(torch.bfloat16), weight.to(torch.bfloat16), bias=bias.to(torch.bfloat16) if bias is not None else None, groups=groups)
        output = torch.relu(output)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_grouped_conv1d_bfloat16_function",
    "inputs": [
        ((1, 64, 128), torch.float32),
        ((32, 64, 3), torch.float32),
        (32, torch.int32),
        ((32,), torch.float32) # Optional bias
    ],
    "outputs": [
        ((1, 32, 128), torch.float32)
    ]
}
