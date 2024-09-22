
import torch

def torch_abs_int8_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the absolute value of an int8 tensor.
    """
    return torch.abs(input_tensor.to(torch.int8))

function_signature = {
    "name": "torch_abs_int8_function",
    "inputs": [
        ((4, 4), torch.int8),
    ],
    "outputs": [
        ((4, 4), torch.int8),
    ]
}
