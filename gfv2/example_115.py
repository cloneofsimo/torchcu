
import torch

def inner_product_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Calculates the inner product of two tensors.
    """
    return torch.sum(input_tensor * weight)

function_signature = {
    "name": "inner_product_function",
    "inputs": [
        ((4,), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
