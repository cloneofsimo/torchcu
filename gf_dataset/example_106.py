
import torch
import torch.nn.functional as F
from torch.fft import rfft, irfft

def torch_conv_with_noise_and_rounding(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                                    noise_scale: float, num_buckets: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Applies a 1D convolution with noise injection, rounding, and bucketing.

    Args:
        input_tensor: Input tensor of shape (batch_size, in_channels, seq_len).
        weight: Convolution kernel of shape (out_channels, in_channels, kernel_size).
        bias: Bias tensor of shape (out_channels).
        noise_scale: Standard deviation of Gaussian noise to inject.
        num_buckets: Number of buckets for bucketing the output.
        dtype: Data type for calculations (bfloat16, float32, etc.).

    Returns:
        A tensor of shape (batch_size, out_channels, seq_len) containing the bucketized output.
    """
    input_tensor = input_tensor.to(dtype)
    weight = weight.to(dtype)
    bias = bias.to(dtype)
    
    # Convolution
    output = F.conv1d(input_tensor, weight, bias=bias, padding="same")
    
    # Noise injection
    output += torch.randn_like(output) * noise_scale

    # Rounding
    output = torch.round(output)

    # Bucketization
    output = torch.bucketize(output, torch.linspace(output.min(), output.max(), num_buckets))
    
    return output.to(torch.int8)

function_signature = {
    "name": "torch_conv_with_noise_and_rounding",
    "inputs": [
        ((1, 10, 100), torch.float32),
        ((5, 10, 5), torch.float32),
        ((5,), torch.float32),
        (1.0, torch.float32),
        (10, torch.int32),
        (torch.bfloat16,)
    ],
    "outputs": [
        ((1, 5, 100), torch.int8),
    ]
}
