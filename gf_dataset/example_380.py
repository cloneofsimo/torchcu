
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioDenoiser(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2):
        super(AudioDenoiser, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv3 = nn.Conv1d(out_channels, in_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = F.celu(self.conv1(x))
        x = F.celu(self.conv2(x))
        x = self.conv3(x)
        return x

def torch_audio_denoising_bf16(noisy_audio: torch.Tensor, clean_audio: torch.Tensor) -> torch.Tensor:
    """
    Denoises audio using a convolutional neural network. 
    
    Args:
        noisy_audio (torch.Tensor): The noisy audio input, shape (batch_size, 1, audio_length).
        clean_audio (torch.Tensor): The clean audio target, shape (batch_size, 1, audio_length).
    
    Returns:
        torch.Tensor: The denoised audio output, shape (batch_size, 1, audio_length).
    """
    denoiser = AudioDenoiser(1, 16)
    denoiser = denoiser.to(torch.bfloat16)
    noisy_audio = noisy_audio.to(torch.bfloat16)
    clean_audio = clean_audio.to(torch.bfloat16)
    denoised_audio = denoiser(noisy_audio)
    loss = F.mse_loss(denoised_audio, clean_audio)
    return (denoised_audio - loss).to(torch.float32)

function_signature = {
    "name": "torch_audio_denoising_bf16",
    "inputs": [
        ((1, 1, 1024), torch.float32), 
        ((1, 1, 1024), torch.float32) 
    ],
    "outputs": [
        ((1, 1, 1024), torch.float32),
    ]
}
