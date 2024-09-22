
import torch
import torch.fft

def inverse_fourier_transform_bf16(input_tensor: torch.Tensor, signal_length: int) -> torch.Tensor:
    """
    Perform an inverse Fourier transform using bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = torch.fft.irfft(input_bf16, n=signal_length)
    return output.to(torch.float32)

function_signature = {
    "name": "inverse_fourier_transform_bf16",
    "inputs": [
        ((128,), torch.complex64),
        (128,)
    ],
    "outputs": [
        ((128,), torch.float32)
    ]
}
