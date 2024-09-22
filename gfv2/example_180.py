
import torch

def kth_value_fft_shift(input_tensor: torch.Tensor, k: int) -> torch.Tensor:
    """
    Calculates the k-th value of the FFT-shifted input tensor along the last dimension.
    """
    input_tensor_int8 = input_tensor.to(torch.int8)
    input_fft = torch.fft.fft(input_tensor_int8, dim=-1)
    input_fft_shifted = torch.fft.fftshift(input_fft, dim=-1)
    kth_value = input_fft_shifted[..., k].real.to(torch.float32)
    return kth_value

function_signature = {
    "name": "kth_value_fft_shift",
    "inputs": [
        ((16, 32), torch.float32),
        (int, int)
    ],
    "outputs": [
        ((16,), torch.float32)
    ]
}
