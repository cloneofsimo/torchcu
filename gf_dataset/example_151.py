
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def audio_denoising_function(noisy_audio: torch.Tensor, filter_weights: torch.Tensor) -> torch.Tensor:
    """
    Applies a filter to denoise audio using a convolution operation.
    """
    with autocast():
        denoised_audio = F.conv1d(noisy_audio.unsqueeze(1), filter_weights)
    return denoised_audio.squeeze(1)

function_signature = {
    "name": "audio_denoising_function",
    "inputs": [
        ((1, 1024), torch.float32),  # Example shape: (batch_size, audio_length)
        ((1, 16, 5), torch.float32)  # Example shape: (out_channels, in_channels, kernel_size)
    ],
    "outputs": [
        ((1, 1024), torch.float32),  # Output shape: (batch_size, audio_length)
    ]
}
