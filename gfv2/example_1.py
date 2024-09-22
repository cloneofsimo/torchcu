
import torch
import torch.nn.functional as F
from torch.fft import rfft, irfft

def torch_conv2d_fft_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a 2D convolution using FFT and int8 quantization.
    """
    input_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    bias_int8 = bias.to(torch.int8)

    # FFT-based convolution
    input_fft = rfft(input_int8, 2, normalized=True)
    weight_fft = rfft(weight_int8, 2, normalized=True)
    output_fft = input_fft * weight_fft
    output_int8 = irfft(output_fft, 2, normalized=True).to(torch.int8)

    # Dequantization and bias addition
    output_float = output_int8.to(torch.float32) / 256.0 + bias_int8.to(torch.float32) / 256.0

    return output_float

function_signature = {
    "name": "torch_conv2d_fft_int8_function",
    "inputs": [
        ((1, 1, 4, 4), torch.float32),
        ((1, 1, 2, 2), torch.float32),
        ((1, 1), torch.float32),
    ],
    "outputs": [
        ((1, 1, 3, 3), torch.float32),
    ]
}
