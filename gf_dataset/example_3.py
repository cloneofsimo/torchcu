
import torch
import torch.fft

def torch_complex_fft_softmax(input_tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Performs a complex FFT, applies a softmax along a specified dimension, and then performs an inverse FFT.
    """
    # Convert input to complex64
    input_complex = input_tensor.to(torch.complex64)

    # Perform FFT
    fft_output = torch.fft.fft(input_complex, dim=-1)

    # Apply softmax along the last dimension
    softmax_output = torch.softmax(fft_output.real * scale, dim=-1)
    softmax_output = softmax_output.to(torch.complex64)

    # Perform inverse FFT
    ifft_output = torch.fft.ifft(softmax_output, dim=-1)

    return ifft_output.real.to(torch.float32)

function_signature = {
    "name": "torch_complex_fft_softmax",
    "inputs": [
        ((4, 4, 16), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((4, 4, 16), torch.float32),
    ]
}
