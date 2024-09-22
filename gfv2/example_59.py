
import torch
import torch.fft

def complex_shift_and_fft(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs an inverse FFT shift (ifftshift) followed by a complex FFT.
    """
    shifted_tensor = torch.fft.ifftshift(input_tensor, dim=(-2, -1))
    output_tensor = torch.fft.fft(shifted_tensor, dim=(-2, -1))
    return output_tensor

function_signature = {
    "name": "complex_shift_and_fft",
    "inputs": [
        ((2, 2, 4, 4), torch.complex64),
    ],
    "outputs": [
        ((2, 2, 4, 4), torch.complex64),
    ]
}
