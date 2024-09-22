
import torch
import torch.nn.functional as F
from cutlass import *

def torch_layer_scaling_decay_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Performs a linear transformation with layer scaling decay, followed by ReLU activation.
    """
    output = F.linear(input_tensor, weight, bias)
    output = output * scale
    return F.relu(output)

function_signature = {
    "name": "torch_layer_scaling_decay_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, ), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
