
import torch

def audio_clipping_hardshrink_fp32(input_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Applies audio clipping followed by hard shrink operation on the input tensor.

    Args:
        input_tensor (torch.Tensor): The input tensor representing audio data.
        threshold (float): The threshold value for hard shrink.

    Returns:
        torch.Tensor: The output tensor after applying audio clipping and hard shrink.
    """
    clipped_tensor = torch.clamp(input_tensor, -1.0, 1.0)  # Audio clipping
    hard_shrinked_tensor = torch.where(torch.abs(clipped_tensor) > threshold, clipped_tensor, torch.zeros_like(clipped_tensor))  # Hard shrink
    return hard_shrinked_tensor.to(torch.float32)  # Return FP32 output

function_signature = {
    "name": "audio_clipping_hardshrink_fp32",
    "inputs": [
        ((1000,), torch.float32),  # Sample audio data
        (torch.float32)  # Threshold value
    ],
    "outputs": [
        ((1000,), torch.float32)  # Output tensor with the same shape
    ]
}
