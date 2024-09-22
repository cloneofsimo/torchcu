
import torch
import torch.nn.functional as F
from cutlass import *

def max_euclidean_distance_gradient_accumulation(input1: torch.Tensor, input2: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Calculates the maximum pairwise Euclidean distance between elements in two tensors after applying a grid sampling operation,
    and accumulates gradients for the grid.

    Args:
        input1: First input tensor (batch_size, channels, height, width).
        input2: Second input tensor (batch_size, channels, height, width).
        grid: Grid tensor (batch_size, height, width, 2).

    Returns:
        A tensor containing the maximum pairwise Euclidean distances.
    """

    # Convert to bfloat16 for faster computation
    input1_bf16 = input1.to(torch.bfloat16)
    input2_bf16 = input2.to(torch.bfloat16)

    # Grid sampling with interpolation
    sampled_input1 = F.grid_sample(input1_bf16, grid, mode='bilinear', align_corners=False)
    sampled_input2 = F.grid_sample(input2_bf16, grid, mode='bilinear', align_corners=False)

    # Calculate pairwise Euclidean distances
    distances = torch.cdist(sampled_input1.view(input1.shape[0], -1), sampled_input2.view(input2.shape[0], -1))

    # Find maximum distance for each batch element
    max_distances = torch.max(distances, dim=1).values

    # Gradient accumulation for grid (requires_grad=True on grid)
    max_distances.backward(retain_graph=True)

    # Return the maximum distances
    return max_distances.to(torch.float32)

function_signature = {
    "name": "max_euclidean_distance_gradient_accumulation",
    "inputs": [
        ((1, 16, 256, 256), torch.float32),
        ((1, 16, 256, 256), torch.float32),
        ((1, 256, 256, 2), torch.float32),
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
