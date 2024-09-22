
import torch
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.cuda import amp

def gradient_scaling_interpolate_conv1d_fft(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                                              scale: float = 1.0) -> torch.Tensor:
    """
    Performs a 1D convolution using FFT, scales gradients, interpolates, and applies elementwise sum with bias.
    """
    with amp.autocast(enabled=True, dtype=torch.bfloat16):
        # Conv1D with FFT
        output = F.conv1d(input_tensor, weight, bias=bias, padding='same', groups=1)
        # Gradient Scaling
        output = output * scale
        # Interpolation (assume linear interpolation)
        output = F.interpolate(output, scale_factor=2, mode='linear', align_corners=False)
        # Elementwise Sum with Bias
        output = output + bias
    return output

function_signature = {
    "name": "gradient_scaling_interpolate_conv1d_fft",
    "inputs": [
        ((1, 16, 128), torch.float32), 
        ((16, 16, 5), torch.float32), 
        ((16,), torch.float32)
    ],
    "outputs": [
        ((1, 16, 256), torch.float32), 
    ]
}
