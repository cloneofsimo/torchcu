
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft

class Conv1dFFTInt8(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, mode='same'):
        super(Conv1dFFTInt8, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.mode = mode

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

        # Quantize weight to int8
        self.weight.data = self.weight.data.int()

    def forward(self, x):
        # Pad input
        x = F.pad(x, (self.padding, self.padding), mode=self.mode)

        # Calculate output shape
        output_size = (x.shape[2] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        # FFT for convolution
        x_f = fft(x, dim=2)
        weight_f = fft(self.weight.float(), dim=2)
        
        # Multiply in frequency domain
        output_f = x_f.unsqueeze(1) * weight_f.unsqueeze(0) 

        # Sum over groups
        output_f = torch.sum(output_f, dim=2)

        # Inverse FFT
        output = ifft(output_f, dim=2).real

        # Apply bias
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        # Crop to desired output size
        output = output[:, :, :output_size]

        return output

def int8_fft_conv1d(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, stride=1, padding=0, dilation=1, groups=1, mode='same') -> torch.Tensor:
    """
    Performs a 1D convolution using FFT with int8 quantized weights.
    """
    # Assuming weight is already int8
    conv = Conv1dFFTInt8(input_tensor.shape[1], weight.shape[0], weight.shape[2], stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, mode=mode)
    conv.weight.data = weight.int()
    if bias is not None:
        conv.bias.data = bias.float()
    output = conv(input_tensor)
    return output

function_signature = {
    "name": "int8_fft_conv1d",
    "inputs": [
        ((1, 3, 10), torch.int8),
        ((2, 3, 3), torch.int8),
        ((2,), torch.float32),
    ],
    "outputs": [
        ((1, 2, 10), torch.float32),
    ]
}
