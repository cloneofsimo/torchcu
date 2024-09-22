import torch
import torch.fft as fft

def fourier_transform(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies the Fourier Transform to the input tensor.

    Args:
        input_tensor (torch.Tensor): The input tensor to apply the Fourier Transform to.

    Returns:
        torch.Tensor: The Fourier Transform of the input tensor.
    """
    # Apply the Fourier Transform
    fft_out = fft.fft2(input_tensor)

    # Shift the Fourier Transform to the center of the tensor
    fft_out = fft.fftshift(fft_out)

    return fft_out



# function_signature
function_signature = {
    "name": "fourier_transform",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [((4, 4), torch.complex64)]
}