
import torch
import torch.nn.functional as F

def torch_audio_upsampling_with_bce(input_tensor: torch.Tensor, target_tensor: torch.Tensor, upsampling_factor: int) -> torch.Tensor:
    """
    Upsamples audio input using a specified factor, then calculates binary cross-entropy loss with the target.

    Args:
        input_tensor (torch.Tensor): Input audio tensor of shape (batch_size, num_channels, num_frames).
        target_tensor (torch.Tensor): Target tensor of the same shape as the upsampled input.
        upsampling_factor (int): Upsampling factor for the audio input.

    Returns:
        torch.Tensor: The calculated binary cross-entropy loss.
    """
    # Upsample input
    upsampled_input = torch.nn.functional.interpolate(input_tensor, scale_factor=upsampling_factor, mode='linear', align_corners=False)
    # Calculate binary cross-entropy loss
    bce_loss = F.binary_cross_entropy(upsampled_input, target_tensor)
    # Return the loss
    return bce_loss

function_signature = {
    "name": "torch_audio_upsampling_with_bce",
    "inputs": [
        ((1, 1, 10), torch.float32),  # Input audio tensor
        ((1, 1, 10), torch.float32),  # Target tensor
        (1, torch.int32)  # Upsampling factor
    ],
    "outputs": [
        ((1,), torch.float32),  # BCE loss
    ]
}
