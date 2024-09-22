
import torch
import torch.nn as nn
from cutlass import *

class Vocoder(nn.Module):
    def __init__(self, n_mels, n_fft, hop_length, win_length, sample_rate):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Generates audio from mel-spectrogram using Griffin-Lim algorithm.

        Args:
            mel_spectrogram (torch.Tensor): Mel-spectrogram of shape (batch_size, n_mels, T).

        Returns:
            torch.Tensor: Reconstructed audio waveform of shape (batch_size, T).
        """
        # Convert mel-spectrogram to linear spectrogram using inverse mel transform
        linear_spectrogram = torch.exp(mel_spectrogram)

        # Apply Griffin-Lim algorithm
        audio_waveform = torch.istft(linear_spectrogram, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, center=True)

        return audio_waveform

function_signature = {
    "name": "vocoder",
    "inputs": [
        ((1, 80, 128), torch.float32)
    ],
    "outputs": [
        ((1, 16384), torch.float32)
    ]
}
