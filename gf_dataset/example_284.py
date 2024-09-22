
import torch
from torch.nn.functional import  harmonic_percussive_separation

def torch_audio_separation_function(audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs harmonic-percussive separation on an audio signal.

    Args:
        audio_tensor (torch.Tensor): Input audio tensor with shape (batch_size, time_steps).

    Returns:
        torch.Tensor: Separated harmonic components of the audio.
    """
    # Convert to bfloat16 for reduced memory usage
    audio_bf16 = audio_tensor.to(torch.bfloat16)
    harmonics, percussive = harmonic_percussive_separation(audio_bf16)
    
    # Convert back to float32 for return
    return harmonics.to(torch.float32)


function_signature = {
    "name": "torch_audio_separation_function",
    "inputs": [
        ((1, 1024), torch.float32)
    ],
    "outputs": [
        ((1, 1024), torch.float32)
    ]
}
