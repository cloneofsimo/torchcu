
import torch
import torchaudio

def audio_spectral_centroid_function(audio_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Calculates the spectral centroid of an audio tensor.

    Args:
        audio_tensor (torch.Tensor): Audio tensor of shape (batch_size, channels, time_steps).
        sample_rate (int): Sampling rate of the audio.

    Returns:
        torch.Tensor: Spectral centroid values of shape (batch_size, channels).
    """
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(audio_tensor)
    spectral_centroid = torchaudio.functional.spectral_centroid(spec, sample_rate)
    return spectral_centroid.mean(dim=-1)

function_signature = {
    "name": "audio_spectral_centroid_function",
    "inputs": [
        ((1, 1, 16000), torch.float32),
        (16000, )
    ],
    "outputs": [
        ((1, 1), torch.float32)
    ]
}
