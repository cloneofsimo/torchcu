
import torch

def torch_matrix_stats(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the mean and determinant of a square matrix.
    """
    mean = torch.mean(input_tensor, dim=1)
    det = torch.det(input_tensor)
    return mean, det

function_signature = {
    "name": "torch_matrix_stats",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4,), torch.float32),
        ((1,), torch.float32)
    ]
}
