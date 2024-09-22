
import torch
import torch.nn.functional as F
from cutlass import *

def torch_exp_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies the exponential function element-wise.
    """
    return torch.exp(input_tensor)

function_signature = {
    "name": "torch_exp_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
