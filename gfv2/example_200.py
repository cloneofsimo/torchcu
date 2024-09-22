
import torch
import torch.fft
import numpy as np

def conv3d_fft_einsum_squeeze_function(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Performs a 3D convolution using FFT, einsum, and squeezing.
    """
    # Pad input to handle convolution without boundary issues
    input_padded = torch.nn.functional.pad(input_tensor, (kernel.shape[2] // 2, kernel.shape[2] // 2, kernel.shape[1] // 2, kernel.shape[1] // 2, kernel.shape[0] // 2, kernel.shape[0] // 2), mode='constant', value=0)

    # Perform 3D FFT on both input and kernel
    input_fft = torch.fft.fftn(input_padded, dim=[-3, -2, -1])
    kernel_fft = torch.fft.fftn(kernel, dim=[-3, -2, -1])

    # Multiply in frequency domain using einsum
    output_fft = torch.einsum('...ijk,ijk->...ijk', input_fft, kernel_fft)

    # Perform inverse FFT to get convolution result
    output = torch.fft.ifftn(output_fft, dim=[-3, -2, -1]).real

    # Squeeze output to remove padding
    output = torch.squeeze(output, dim=[-3, -2, -1])

    # Convert to fp32 and return
    return output.to(torch.float32)

function_signature = {
    "name": "conv3d_fft_einsum_squeeze_function",
    "inputs": [
        ((1, 1, 3, 3, 3), torch.int8),
        ((1, 1, 2, 2, 2), torch.int8)
    ],
    "outputs": [
        ((1, 1, 2, 2, 2), torch.float32)
    ]
}
