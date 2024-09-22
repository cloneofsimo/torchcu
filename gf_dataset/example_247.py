
import torch
import torch.nn.functional as F

def bilateral_filter_threshold(input_tensor: torch.Tensor, kernel_size: int, sigma_color: float, sigma_space: float, threshold: float) -> torch.Tensor:
    """
    Applies a bilateral filter to the input tensor, then thresholds the result.
    """
    filtered = F.bilateral_filter(input_tensor, kernel_size, sigma_color, sigma_space)
    thresholded = torch.where(filtered > threshold, filtered, torch.zeros_like(filtered))
    return thresholded.clamp(min=0.0, max=1.0)

function_signature = {
    "name": "bilateral_filter_threshold",
    "inputs": [
        ((16, 16, 3), torch.float32),
        (11, torch.int32),
        (1.0, torch.float32),
        (1.0, torch.float32),
        (0.5, torch.float32),
    ],
    "outputs": [
        ((16, 16, 3), torch.float32),
    ]
}
