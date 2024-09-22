
import torch

def grid_sampler_function(input_tensor: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Perform grid sampling using an affine transformation.
    """
    grid = torch.nn.functional.affine_grid(theta, input_tensor.size())
    output = torch.nn.functional.grid_sample(input_tensor, grid, align_corners=False)
    return output

function_signature = {
    "name": "grid_sampler_function",
    "inputs": [
        ((4, 3, 224, 224), torch.float32),
        ((1, 6), torch.float32)
    ],
    "outputs": [
        ((4, 3, 224, 224), torch.float32)
    ]
}
