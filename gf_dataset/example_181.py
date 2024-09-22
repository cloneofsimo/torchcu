
import torch
import torch.nn.functional as F
from torch.nn import GroupNorm

def torch_audio_feature_extraction(audio_tensor: torch.Tensor, sample_rate: int, window_size: int, hop_length: int) -> torch.Tensor:
    """
    Extracts audio features from an audio tensor, including spectral centroid and interpolation.

    Args:
        audio_tensor (torch.Tensor): The audio tensor, expected to be in shape (batch_size, time_steps).
        sample_rate (int): The sample rate of the audio.
        window_size (int): The window size for the STFT (in samples).
        hop_length (int): The hop length for the STFT (in samples).

    Returns:
        torch.Tensor: A tensor containing the extracted spectral centroids, shape (batch_size, time_steps).
    """
    # STFT
    specgram = torch.stft(audio_tensor, n_fft=window_size, hop_length=hop_length, return_complex=True)
    magnitudes = torch.abs(specgram)

    # Spectral Centroid
    centroid = torch.sum(magnitudes * torch.arange(magnitudes.shape[1], dtype=torch.float32, device=magnitudes.device), dim=1)
    centroid = centroid / torch.sum(magnitudes, dim=1) 
    centroid = centroid * sample_rate / window_size 

    # Interpolation
    interpolated_features = F.interpolate(centroid.unsqueeze(1), size=audio_tensor.shape[1], mode='linear', align_corners=False).squeeze(1)

    # Group Normalization
    norm_features = GroupNorm(num_groups=4, num_channels=1)(interpolated_features.unsqueeze(1)).squeeze(1)

    return norm_features


function_signature = {
    "name": "torch_audio_feature_extraction",
    "inputs": [
        ((1, 1024), torch.float32),
        (1, torch.int32),
        (1, torch.int32),
        (1, torch.int32)
    ],
    "outputs": [
        ((1, 1024), torch.float32),
    ]
}
