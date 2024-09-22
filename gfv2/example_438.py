
import torch

def my_function(input_tensor: torch.Tensor,  min_value: float) -> torch.Tensor:
    """
    Generates a tensor of the same size as the input tensor, filled with random numbers 
    uniformly distributed between min_value and 1.0.
    """
    return torch.min(torch.rand_like(input_tensor), torch.full_like(input_tensor, min_value))

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1, 1), torch.float32),
        (torch.float32,)
    ],
    "outputs": [
        ((1, 1), torch.float32),
    ]
}
