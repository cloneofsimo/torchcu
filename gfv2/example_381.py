
import torch

def pixel_shuffle_unsqueeze(input_tensor: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    """
    Performs pixel shuffle and unsqueeze operation on the input tensor.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape (B, C, H, W).
        upscale_factor (int): Upscale factor for pixel shuffle.

    Returns:
        torch.Tensor: Output tensor with shape (B, C * upscale_factor ** 2, H * upscale_factor, W * upscale_factor).
    """
    output = torch.nn.functional.pixel_shuffle(input_tensor, upscale_factor=upscale_factor)
    output = output.unsqueeze(1)
    return output

function_signature = {
    "name": "pixel_shuffle_unsqueeze",
    "inputs": [
        ((1, 3, 4, 4), torch.float32),
        (int)
    ],
    "outputs": [
        ((1, 1, 8, 8), torch.float32)
    ]
}
