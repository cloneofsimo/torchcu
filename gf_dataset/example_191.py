
import torch
import torch.nn as nn

class AudioDenoiser(nn.Module):
    def __init__(self, channels=128, kernel_size=3, groups=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, channels, kernel_size, padding=kernel_size // 2)
        self.group_norm1 = nn.GroupNorm(groups, channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.group_norm2 = nn.GroupNorm(groups, channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(channels, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        x = self.conv1(audio_input)
        x = self.group_norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.group_norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        return x


def torch_audio_denoiser_fp16(audio_input: torch.Tensor) -> torch.Tensor:
    """
    Performs audio denoising using a convolutional neural network with group normalization.
    Returns denoised audio in FP16.
    """
    model = AudioDenoiser().to(torch.float16)
    with torch.cuda.amp.autocast():
        denoised_audio = model(audio_input.to(torch.float16))
    return denoised_audio.to(torch.float16)

function_signature = {
    "name": "torch_audio_denoiser_fp16",
    "inputs": [
        ((1, 16000), torch.float32),
    ],
    "outputs": [
        ((1, 16000), torch.float16),
    ]
}

