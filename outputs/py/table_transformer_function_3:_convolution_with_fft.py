import torch
import torch.fft as fft

def convolution_with_fft(input_tensor: torch.Tensor, kernel_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies a convolution to the input tensor using the Fast Fourier Transform (FFT).

    Args:
        input_tensor (torch.Tensor): The input tensor to apply the convolution to.
        kernel_tensor (torch.Tensor): The kernel to use for the convolution.

    Returns:
        torch.Tensor: The output of the convolution.
    """
    # Apply the Fourier Transform to the input tensor and kernel
    fft_input = fft.fft2(input_tensor)
    fft_kernel = fft.fft2(kernel_tensor)

    # Multiply the Fourier Transforms
    fft_out = fft_input * fft_kernel

    # Apply the inverse Fourier Transform
    out = fft.ifft2(fft_out)

    # Return the real part of the output
    return out.real



# function_signature
function_signature = {
    "name": "convolution_with_fft",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [((4, 4), torch.float32)]
}