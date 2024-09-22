
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegularizedGridSample(nn.Module):
    def __init__(self, scale=1.0):
        super(RegularizedGridSample, self).__init__()
        self.scale = scale

    def forward(self, input_tensor: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        Performs grid sampling with regularization and layer scaling.

        Args:
            input_tensor: The input tensor to be sampled, shape (B, C, H, W).
            grid: The sampling grid, shape (B, H, W, 2).

        Returns:
            The sampled output tensor, shape (B, C, H, W).
        """
        # Regularize the grid to prevent out-of-bounds access
        grid = grid.clamp(min=-1.0, max=1.0)

        # Perform grid sampling using F.grid_sample
        sampled_output = F.grid_sample(input_tensor, grid, align_corners=False, mode='bilinear')

        # Apply layer scaling
        return sampled_output * self.scale

function_signature = {
    "name": "regularized_grid_sample",
    "inputs": [
        ((1, 3, 128, 128), torch.float32),  # Input tensor
        ((1, 128, 128, 2), torch.float32)  # Grid
    ],
    "outputs": [
        ((1, 3, 128, 128), torch.float32)  # Sampled output
    ]
}

