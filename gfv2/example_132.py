
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

def logsumexp_grad_checkpointing(input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the logsumexp along a specific dimension, using gradient checkpointing for efficiency.
    """
    def logsumexp_inner(input_tensor):
        return torch.logsumexp(input_tensor, dim=dim)

    output = checkpoint(logsumexp_inner, input_tensor)
    output.backward(torch.ones_like(output))
    return output

function_signature = {
    "name": "logsumexp_grad_checkpointing",
    "inputs": [
        ((2, 3, 4), torch.float32),
        (0, torch.int32),
    ],
    "outputs": [
        ((2, 4), torch.float32),
    ]
}

