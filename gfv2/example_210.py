
import torch

def frobenius_norm_selu_inplace(input_tensor: torch.Tensor, scale: float = 1.0507009873554805, alpha: float = 1.673263242354377) -> torch.Tensor:
    """
    Calculates the Frobenius norm of a tensor, applies SELU activation inplace, and scales the result.
    """
    norm = torch.linalg.norm(input_tensor, ord='fro')
    input_tensor.mul_(scale * torch.selu(input_tensor, alpha=alpha))
    return norm

function_signature = {
    "name": "frobenius_norm_selu_inplace",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
