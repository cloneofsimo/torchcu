
import torch
import torch.nn.functional as F
from torch.fft import rfft, irfft

class SoftShrinkConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, lambda_value=0.5):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.lambda_value = lambda_value

    def forward(self, x):
        # Convolution
        y = self.conv(x)

        # Soft shrinkage
        y = torch.where(y.abs() > self.lambda_value, y - self.lambda_value * y.sign(), torch.zeros_like(y))

        # Clamp
        y = torch.clamp(y, -1, 1)

        # FFT Convolution
        y = rfft(y, dim=2, norm="ortho")
        y = rfft(y, dim=3, norm="ortho")
        return irfft(y, dim=3, norm="ortho")

function_signature = {
    "name": "soft_shrink_conv2d_fft",
    "inputs": [
        ((1, 3, 32, 32), torch.float32),
    ],
    "outputs": [
        ((1, 8, 32, 32), torch.float32),
    ]
}
