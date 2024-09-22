
import torch
import torch.fft
from torch import Tensor

def istft_example(input_tensor: Tensor, input_length: int, hop_length: int,
                  win_length: int, window: Tensor) -> Tensor:
    """
    Performs an inverse short-time Fourier transform (ISTFT).

    Args:
        input_tensor: The input complex tensor, shape (batch_size, n_fft//2+1, n_frames)
        input_length: The length of the original signal (before STFT)
        hop_length: Hop length used in the STFT
        win_length: Window length used in the STFT
        window: The window function used in the STFT

    Returns:
        The reconstructed signal, shape (batch_size, input_length)
    """
    # Convert to FP16 for potential performance improvement
    input_tensor_fp16 = input_tensor.to(torch.float16)
    window_fp16 = window.to(torch.float16)

    # Perform ISTFT
    output = torch.fft.istft(
        input_tensor_fp16, n_fft=input_tensor.size(1) * 2 - 1, hop_length=hop_length,
        win_length=win_length, window=window_fp16, length=input_length,
        center=True, normalized=False
    )

    # Convert back to FP32 for output
    return output.to(torch.float32)

function_signature = {
    "name": "istft_example",
    "inputs": [
        ((8, 257, 100), torch.complex64),
        (1000,), torch.int32,
        (256,), torch.int32,
        (512,), torch.int32,
        ((512,), torch.float32)
    ],
    "outputs": [
        ((8, 1000), torch.float32)
    ]
}
