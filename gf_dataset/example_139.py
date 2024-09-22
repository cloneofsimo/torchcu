
import torch
import torch.nn.functional as F

def torch_affine_grid_function(input_tensor: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Generates a 2D sampling grid based on affine transformation parameters.
    """
    grid = F.affine_grid(theta, input_tensor.size(), align_corners=False)
    return grid

function_signature = {
    "name": "torch_affine_grid_function",
    "inputs": [
        ((1, 1, 10, 10), torch.float32),
        ((1, 2, 3), torch.float32)
    ],
    "outputs": [
        ((1, 1, 10, 10, 2), torch.float32)
    ]
}
