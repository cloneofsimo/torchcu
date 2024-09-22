
import torch
import torch.fft

def conv_fft_diagflat_inplace(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs convolution using FFT and then applies a diagonal matrix operation.
    The result is then stored back in the input tensor (inplace).
    """
    # Convert to complex tensors
    input_complex = input_tensor.to(torch.complex64)
    weight_complex = weight.to(torch.complex64)

    # FFT of input tensor
    input_fft = torch.fft.fft(input_complex, dim=-1)

    # Apply diagonal matrix operation
    diagonal_matrix = torch.diagflat(weight_complex)
    output_fft = torch.matmul(input_fft, diagonal_matrix)

    # Inverse FFT
    output_complex = torch.fft.ifft(output_fft, dim=-1)

    # Store the result back in the input tensor (inplace)
    input_tensor.copy_(output_complex.real)

    return input_tensor

function_signature = {
    "name": "conv_fft_diagflat_inplace",
    "inputs": [
        ((1, 2, 3, 4), torch.float32),
        ((4, ), torch.float32)
    ],
    "outputs": [
        ((1, 2, 3, 4), torch.float32)
    ]
}

