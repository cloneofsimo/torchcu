
import torch
import torch.nn.functional as F
import numpy as np

def spectrogram_conv_elu_fp32_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Applies a spectrogram, convolution, and ELU activation to the input tensor.
    """
    # Spectrogram calculation
    spectrogram = torch.stft(input_tensor, n_fft=256, hop_length=128, win_length=256)
    spectrogram = torch.abs(spectrogram) ** 2

    # Convolution
    output = F.conv_tbc(spectrogram, weight, bias)

    # ELU activation
    output = F.elu(output)

    return output


function_signature = {
    "name": "spectrogram_conv_elu_fp32_function",
    "inputs": [
        ((1, 1, 16000), torch.float32),
        ((128, 1, 128), torch.float32),
        ((128,), torch.float32)
    ],
    "outputs": [
        ((1, 128, 128), torch.float32)
    ]
}
