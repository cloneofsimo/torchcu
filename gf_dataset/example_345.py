
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

def mel_spectrogram_cross_fade(audio_tensor: torch.Tensor, teacher_output: torch.Tensor, cross_fade_ratio: float) -> torch.Tensor:
    """
    Calculate Mel Spectrogram and apply cross-fading with teacher output for adversarial training.

    Args:
        audio_tensor: Input audio tensor (batch_size, num_frames).
        teacher_output: Output tensor from the teacher model (batch_size, num_features).
        cross_fade_ratio: Ratio for cross-fading (0.0 for pure teacher, 1.0 for pure student).

    Returns:
        Mel spectrogram with cross-fading applied.
    """
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=1024, hop_length=512, n_mels=80)(audio_tensor)
    mel_spectrogram = mel_spectrogram.mean(dim=1)  # Average over time dimension
    
    if cross_fade_ratio > 0:
        mel_spectrogram = (1 - cross_fade_ratio) * mel_spectrogram + cross_fade_ratio * teacher_output
    return mel_spectrogram

function_signature = {
    "name": "mel_spectrogram_cross_fade",
    "inputs": [
        ((1000,), torch.float32),
        ((80,), torch.float32),
        ((), torch.float32),
    ],
    "outputs": [
        ((80,), torch.float32),
    ]
}
