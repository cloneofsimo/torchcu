
import torch

def torch_ceil_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the ceiling of each element in the input tensor.
    """
    return torch.ceil(input_tensor)

function_signature = {
    "name": "torch_ceil_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
