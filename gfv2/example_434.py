
import torch

def softmin_function(input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Calculates the softmin of a tensor along a given dimension.
    """
    return torch.softmax(-input_tensor, dim=dim)

function_signature = {
    "name": "softmin_function",
    "inputs": [
        ((3, 4), torch.float32),
        (int,)
    ],
    "outputs": [
        ((3, 4), torch.float32),
    ]
}
