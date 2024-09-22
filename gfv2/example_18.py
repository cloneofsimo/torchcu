
import torch
import torch.fft
from torch.fft import rfft, irfft

def torch_double_linear_hilbert_inplace(input_tensor: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor) -> torch.Tensor:
    """
    Performs a double linear transformation with Hilbert transform in between, modifies the input tensor inplace.
    """
    input_tensor = input_tensor.to(torch.complex64)
    output = torch.matmul(input_tensor, weight1.t())
    output = torch.fft.fft(output, dim=1)
    output[:, :output.size(1) // 2] = 0.0 + 0.0j  # Zero out negative frequencies
    output = torch.fft.ifft(output, dim=1)
    output = torch.matmul(output, weight2.t())
    output = output.to(torch.float32)
    input_tensor.copy_(output)
    return input_tensor


function_signature = {
    "name": "torch_double_linear_hilbert_inplace",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
