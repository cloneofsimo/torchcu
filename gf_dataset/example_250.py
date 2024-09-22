
import torch
import torch.fft
from torch.nn.functional import mish

def torch_idft_mish_fp32_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the inverse discrete Fourier transform (IDFT) followed by the Mish activation function.
    """
    output = torch.fft.irfft(input_tensor, 1, normalized=True)  # IDFT
    return mish(output, inplace=True)  # Mish activation (inplace)

function_signature = {
    "name": "torch_idft_mish_fp32_function",
    "inputs": [
        ((16, 16), torch.complex64)
    ],
    "outputs": [
        ((16, 16), torch.float32),
    ]
}
