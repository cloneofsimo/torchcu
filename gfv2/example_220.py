
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft, irfft

class Conv3dFFT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv3dFFT, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))

    def forward(self, x):
        # Pad input
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding, self.padding, self.padding))
        # Perform FFT on input
        x_fft = rfft(x, dim=(-3, -2, -1))
        # Perform convolution in frequency domain
        weight_fft = rfft(self.weight, dim=(-3, -2, -1))
        out_fft = x_fft * weight_fft
        # Perform inverse FFT
        out = irfft(out_fft, dim=(-3, -2, -1))
        # Crop to original size
        out = out[:, :, self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding]
        # Apply stride
        out = out[:, :, ::self.stride, ::self.stride, ::self.stride]
        return out

def my_function(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Performs 3D convolution using FFT, applies element-wise comparison,
    and returns the result.
    """
    
    # Apply 3D convolution using FFT
    conv_module = Conv3dFFT(in_channels=weights.shape[1], out_channels=weights.shape[0], kernel_size=weights.shape[2:])
    output = conv_module(input_tensor.float())
    
    # Generate random tensor for comparison
    comparison_tensor = torch.rand_like(output, dtype=torch.float32)
    
    # Element-wise comparison (greater than)
    output = torch.where(output > comparison_tensor, output, torch.zeros_like(output))
    
    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((10, 4, 8, 8, 8), torch.float32),
        ((4, 4, 3, 3, 3), torch.float32)
    ],
    "outputs": [
        ((10, 4, 8, 8, 8), torch.float32),
    ]
}
