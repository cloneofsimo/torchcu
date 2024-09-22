
import torch

def inner_product_inplace(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs an inner product operation inplace, modifying the input tensor.
    """
    input_tensor.mul_(weight)
    return input_tensor

function_signature = {
    "name": "inner_product_inplace",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
