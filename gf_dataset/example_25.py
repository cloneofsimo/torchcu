
import torch
import torch.fft

def torch_fft_mish_distance(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on input and target tensors:
    1. Applies FFT shift to both tensors.
    2. Computes Mish activation on both tensors.
    3. Calculates pairwise Chebyshev distance between shifted and activated tensors.
    4. Returns the distance tensor.
    """
    input_shifted = torch.fft.fftshift(input_tensor, dim=-1)
    target_shifted = torch.fft.fftshift(target_tensor, dim=-1)

    input_mish = torch.mish(input_shifted)
    target_mish = torch.mish(target_shifted)

    distance = torch.cdist(input_mish, target_mish, p=float('inf'))
    return distance

function_signature = {
    "name": "torch_fft_mish_distance",
    "inputs": [
        ((4, 4), torch.float32), 
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
