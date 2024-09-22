
import torch
import torch.fft

def torch_ifft_fp32_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs an inverse Fourier transform on a complex-valued tensor.
    """
    return torch.fft.ifft(input_tensor, dim=-1).real.to(torch.float32)

function_signature = {
    "name": "torch_ifft_fp32_function",
    "inputs": [
        ((4, 4), torch.complex64)  # Assuming complex input
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
