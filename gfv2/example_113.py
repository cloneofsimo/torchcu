
import torch
import torch.nn as nn

def log_filter_affine_grid_generator(input_tensor: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Applies log filter to input tensor, generates affine grid based on theta, and returns the transformed grid.
    """
    # Log Filter
    input_tensor = torch.log(input_tensor + 1e-6)

    # Affine Grid Generator
    grid = nn.functional.affine_grid(theta, input_tensor.size())

    return grid

function_signature = {
    "name": "log_filter_affine_grid_generator",
    "inputs": [
        ((2, 3, 224, 224), torch.float32),
        ((2, 6), torch.float32),
    ],
    "outputs": [
        ((2, 3, 224, 224), torch.float32),
    ]
}
