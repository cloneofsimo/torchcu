
import torch

def gather_function(input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Gathers elements from input tensor based on indices.
    """
    return torch.gather(input_tensor, dim=1, index=indices)

function_signature = {
    "name": "gather_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.int64),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
