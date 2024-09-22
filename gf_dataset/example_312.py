
import torch
import torch.nn as nn
import torchaudio
import numpy as np

class MFCC_Erosion(nn.Module):
    def __init__(self, n_mfcc=13, n_fft=1024, hop_length=512, erosion_kernel_size=3):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.erosion_kernel_size = erosion_kernel_size
        self.mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)

    def forward(self, audio_signal: torch.Tensor, seed: int) -> torch.Tensor:
        """
        Extracts MFCCs and performs morphological erosion on them.

        Args:
            audio_signal (torch.Tensor): Audio signal with shape (batch_size, 1, time_steps)
            seed (int): Seed for manual seed in erosion.

        Returns:
            torch.Tensor: Eroded MFCCs with shape (batch_size, n_mfcc, time_steps)
        """
        torch.manual_seed(seed)
        mfccs = self.mfcc_transform(audio_signal)
        eroded_mfccs = torch.Tensor(mfccs.shape)
        for i in range(mfccs.shape[0]):
            for j in range(mfccs.shape[1]):
                eroded_mfccs[i, j] = torch.nn.functional.max_pool1d(
                    mfccs[i, j].unsqueeze(0), 
                    kernel_size=self.erosion_kernel_size,
                    stride=1,
                    padding=self.erosion_kernel_size // 2
                ).squeeze(0)
        return eroded_mfccs

function_signature = {
    "name": "mfcc_erosion_forward",
    "inputs": [
        ((1, 1, 16000), torch.float32),
        ((), torch.int32),
    ],
    "outputs": [
        ((1, 13, 8001), torch.float32),
    ]
}
