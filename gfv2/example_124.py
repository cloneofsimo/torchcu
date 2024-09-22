
import torch
import torch.fft

def inverse_fourier_transform_fp32(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs inverse Fourier transform on a complex-valued tensor.

    Args:
        input_tensor (torch.Tensor): Complex-valued tensor with shape (batch_size, channels, height, width) 
                                  or (batch_size, channels, sequence_length) representing frequency domain data.

    Returns:
        torch.Tensor: Real-valued tensor with the same shape as the input, representing the time domain signal.
    """
    return torch.fft.irfft(input_tensor, signal_ndim=input_tensor.ndim - 2).real.to(torch.float32)

function_signature = {
    "name": "inverse_fourier_transform_fp32",
    "inputs": [
        ((2, 3, 8, 8), torch.complex64),
    ],
    "outputs": [
        ((2, 3, 8, 8), torch.float32),
    ]
}
