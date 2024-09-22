
import torch
import torch.fft

def torch_fft_shift_divide(input_tensor: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    """
    Performs a complex FFT, shifts the frequency spectrum, divides by a divisor, and returns the result.
    """
    fft = torch.fft.fft(input_tensor, dim=-1)
    shifted_fft = torch.fft.fftshift(fft, dim=-1)
    divided_fft = shifted_fft / divisor
    return torch.fft.ifft(torch.fft.ifftshift(divided_fft, dim=-1), dim=-1)

function_signature = {
    "name": "torch_fft_shift_divide",
    "inputs": [
        ((4, 4), torch.complex64),
        ((4, ), torch.complex64)
    ],
    "outputs": [
        ((4, 4), torch.complex64)
    ]
}
