
import torch
import torch.nn.functional as F

def torch_interpolate_mean_function(input_tensor: torch.Tensor, size: int, mode: str = 'bilinear', align_corners: bool = False) -> torch.Tensor:
    """
    Performs interpolation and calculates the mean of the resulting tensor.
    """
    interpolated = F.interpolate(input_tensor, size=size, mode=mode, align_corners=align_corners)
    return torch.mean(interpolated)

function_signature = {
    "name": "torch_interpolate_mean_function",
    "inputs": [
        ((4, 3, 10, 10), torch.float32),
        (10, torch.int32),
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
