
import torch

def subtract_tensor_function(input_tensor: torch.Tensor, value: float) -> torch.Tensor:
    """
    Subtracts a scalar value from each element of a tensor.
    """
    return input_tensor - value

function_signature = {
    "name": "subtract_tensor_function",
    "inputs": [
        ((2, 3), torch.float32),
        (torch.float32,)
    ],
    "outputs": [
        ((2, 3), torch.float32),
    ]
}
