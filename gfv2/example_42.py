
import torch
import torch.nn.functional as F
from cutlass import *

def subtract_and_clamp(input_tensor: torch.Tensor, subtrahend: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Subtract two tensors and clamp the result to a specified range.
    """
    result = torch.sub(input_tensor, subtrahend)
    result = torch.clamp(result, min_val, max_val)
    return result

function_signature = {
    "name": "subtract_and_clamp",
    "inputs": [
        ((16, 16), torch.float32),
        ((16, 16), torch.float32),
        (torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((16, 16), torch.float32),
    ]
}
