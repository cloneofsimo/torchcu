
import torch

def outer_product_int8(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the outer product of a tensor with itself, returning the result as int8.
    """
    output = torch.outer(input_tensor, input_tensor)
    return output.to(torch.int8)

function_signature = {
    "name": "outer_product_int8",
    "inputs": [
        ((4,), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.int8),
    ]
}
