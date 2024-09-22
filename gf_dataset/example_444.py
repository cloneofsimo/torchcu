
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def torch_canny_edge_detection_fp16(image: torch.Tensor, low_threshold: float, high_threshold: float) -> torch.Tensor:
    """
    Applies Canny edge detection to an image, using FP16 for faster computation.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        low_threshold (float): Lower threshold for hysteresis.
        high_threshold (float): Higher threshold for hysteresis.

    Returns:
        torch.Tensor: Edge map tensor of shape (1, H, W).
    """
    with autocast():
        edges = F.canny(image.to(torch.float16), low_threshold, high_threshold)
    return edges.to(torch.float32)

function_signature = {
    "name": "torch_canny_edge_detection_fp16",
    "inputs": [
        ((3, 256, 256), torch.float32),
        (float, ),
        (float, )
    ],
    "outputs": [
        ((1, 256, 256), torch.float32),
    ]
}
