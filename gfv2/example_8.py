
import torch

def cholesky_decomposition(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Cholesky decomposition of a symmetric positive-definite matrix.
    """
    return torch.linalg.cholesky(input_tensor)

function_signature = {
    "name": "cholesky_decomposition",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
