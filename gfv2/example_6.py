
import torch
import torch.nn.functional as F

def torch_affine_grid_generator_fp16(input_tensor: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Generates a 2D affine grid for image warping.

    Args:
        input_tensor (torch.Tensor): The input tensor with shape (B, C, H, W).
        theta (torch.Tensor): The transformation matrix with shape (B, 2, 3).

    Returns:
        torch.Tensor: The affine grid with shape (B, H, W, 2) in fp16.
    """
    grid = F.affine_grid(theta.to(torch.float32), input_tensor.size(), align_corners=False)
    return grid.to(torch.float16)

function_signature = {
    "name": "torch_affine_grid_generator_fp16",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((1, 2, 3), torch.float32)
    ],
    "outputs": [
        ((1, 224, 224, 2), torch.float16),
    ]
}
