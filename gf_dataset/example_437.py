
import torch

def torch_diagflat_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes a 2D tensor with the elements of input on the diagonal.
    """
    return torch.diagflat(input_tensor)

function_signature = {
    "name": "torch_diagflat_function",
    "inputs": [
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
