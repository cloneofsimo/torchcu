
import torch

def torch_det_cudnn_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the determinant of a square matrix using cuDNN.
    Returns a single value representing the determinant.
    """
    return torch.det(input_tensor)

function_signature = {
    "name": "torch_det_cudnn_function",
    "inputs": [
        ((2, 2), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
