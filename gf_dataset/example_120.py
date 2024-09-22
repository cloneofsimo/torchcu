
import torch

def torch_fft_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a forward Fourier transform, matrix multiplication, and inverse Fourier transform.
    """
    # Forward Fourier Transform
    fft_input = torch.fft.fft(input_tensor)

    # Create Identity Matrix
    identity_matrix = torch.eye(input_tensor.shape[-1], dtype=torch.complex64)

    # Matrix Multiplication
    output = torch.matmul(fft_input, identity_matrix)

    # Inverse Fourier Transform
    output = torch.fft.ifft(output)

    # Return the real part (as we're dealing with real-valued input)
    return output.real

function_signature = {
    "name": "torch_fft_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
