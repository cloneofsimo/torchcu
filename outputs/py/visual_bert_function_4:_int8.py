import torch
import numpy as np

def int8(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts the input tensor to int8 data type.

    Args:
    input_tensor (torch.Tensor): The input tensor.

    Returns:
    torch.Tensor: The tensor with int8 data type.
    """
    return torch.clamp(input_tensor, -128, 127).to(torch.int8)



# function_signature
function_signature = {
    "name": "int8",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [((4, 4), torch.int8)]
}