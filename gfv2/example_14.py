
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft, irfft

class HilbertTransformWithNoise(nn.Module):
    def __init__(self, noise_scale=0.1, pool_kernel_size=3):
        super(HilbertTransformWithNoise, self).__init__()
        self.noise_scale = noise_scale
        self.pool_kernel_size = pool_kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size, stride=1)

    def forward(self, x):
        # 1. Hilbert Transform
        x_fft = rfft(x, dim=1)
        x_fft[:, 1:] = 0j  # Set imaginary part to 0 for real signal
        x_hilbert = irfft(x_fft, dim=1)

        # 2. Noise Injection
        noise = torch.randn_like(x_hilbert) * self.noise_scale
        x_hilbert = x_hilbert + noise

        # 3. Mean along last dimension
        x_hilbert = torch.mean(x_hilbert, dim=2)

        # 4. Average Pooling
        x_hilbert = self.avg_pool(x_hilbert.unsqueeze(1)).squeeze(1)

        return x_hilbert

function_signature = {
    "name": "hilbert_transform_with_noise",
    "inputs": [
        ((10, 512, 10), torch.float32)
    ],
    "outputs": [
        ((10, 512), torch.float32)
    ]
}
