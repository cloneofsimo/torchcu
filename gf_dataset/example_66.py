
import torch
import torch.nn.functional as F

def torch_affine_grid_generator_fp16(input_tensor: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Generates a 2D sampling grid using affine transformations.
    """
    input_tensor = input_tensor.to(torch.float16)
    theta = theta.to(torch.float16)
    grid = F.affine_grid(theta, input_tensor.size(), align_corners=False)
    return grid.to(torch.float16)

function_signature = {
    "name": "torch_affine_grid_generator_fp16",
    "inputs": [
        ((1, 1, 5, 5), torch.float32),
        ((1, 2, 3), torch.float32)
    ],
    "outputs": [
        ((1, 1, 5, 5, 2), torch.float16),
    ]
}
