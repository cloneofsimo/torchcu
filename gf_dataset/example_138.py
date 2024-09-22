
import torch
import torch.fft

def torch_conv_ifft_function(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Performs a convolution followed by an inverse FFT.
    """
    output = torch.nn.functional.conv1d(input_tensor, kernel, padding='same')
    output = torch.fft.irfft(output, dim=1)  # Inverse FFT along the second dimension (frequency)
    return output

function_signature = {
    "name": "torch_conv_ifft_function",
    "inputs": [
        ((16, 128), torch.complex64),
        ((1, 32), torch.complex64)
    ],
    "outputs": [
        ((16, 128), torch.float32),
    ]
}
