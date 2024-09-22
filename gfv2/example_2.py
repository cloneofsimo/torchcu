
import torch
import torch.nn.functional as F
from torch.fft import fft, ifft
from scipy.ndimage import binary_closing

def torch_closing_conv_ifft(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Perform a morphological closing operation on the input tensor followed by a convolution and inverse FFT.
    """
    # Convert to binary for morphological closing
    binary_tensor = (input_tensor > 0).float()

    # Perform morphological closing using SciPy
    closing = binary_closing(binary_tensor.cpu().numpy())
    closing_tensor = torch.from_numpy(closing).to(input_tensor.device)

    # Convolution using FFT (frequency domain)
    input_fft = fft(closing_tensor)
    kernel_fft = fft(kernel)
    output_fft = input_fft * kernel_fft
    output = ifft(output_fft)

    # Return real part of the inverse FFT
    return output.real

function_signature = {
    "name": "torch_closing_conv_ifft",
    "inputs": [
        ((16, 16, 16), torch.float32),  # Example shape
        ((3, 3, 3), torch.float32),   # Example shape
    ],
    "outputs": [
        ((16, 16, 16), torch.float32),
    ]
}
