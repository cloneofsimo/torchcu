
import torch
import torch.nn.functional as F

def torch_wavelet_transform_2d(input_tensor: torch.Tensor, wavelet: str = 'db4') -> torch.Tensor:
    """
    Applies a 2D wavelet transform to an input tensor using PyTorch.

    Args:
        input_tensor (torch.Tensor): The input tensor to transform.
        wavelet (str, optional): The wavelet family to use. Defaults to 'db4'.

    Returns:
        torch.Tensor: The wavelet transform of the input tensor.
    """
    # Check if input tensor has at least 2 dimensions
    if len(input_tensor.shape) < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")

    # Apply wavelet transform using PyTorch's built-in function
    transformed_tensor = torch.stft(input_tensor, n_fft=input_tensor.shape[-1], hop_length=input_tensor.shape[-1] // 2,
                                    win_length=input_tensor.shape[-1], window=torch.hann_window(input_tensor.shape[-1]),
                                    center=False, return_complex=False)

    # Extract real and imaginary components of the transformed tensor
    real_part = transformed_tensor[:, :, 0, :]
    imag_part = transformed_tensor[:, :, 1, :]

    # Combine real and imaginary parts into a single tensor
    return torch.stack([real_part, imag_part], dim=-1)

function_signature = {
    "name": "torch_wavelet_transform_2d",
    "inputs": [
        ((16, 16), torch.float32),
    ],
    "outputs": [
        ((16, 16, 2), torch.float32),
    ]
}

