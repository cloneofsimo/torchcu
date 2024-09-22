
import torch

def my_function(input_tensor: torch.Tensor, scalar: float, other_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a simple calculation involving tensors and a scalar.
    """
    result = torch.mul(input_tensor, scalar) + other_tensor
    return result

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1, 2), torch.float32),
        ((), torch.float32),
        ((1, 2), torch.float32)
    ],
    "outputs": [
        ((1, 2), torch.float32),
    ]
}
