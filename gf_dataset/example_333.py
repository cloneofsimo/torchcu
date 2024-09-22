
import torch
import torch.nn.functional as F

def torch_diag_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the diagonal of a square matrix and returns it as a vector.
    """
    return torch.diag(input_tensor)

function_signature = {
    "name": "torch_diag_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4,), torch.float32)
    ]
}
