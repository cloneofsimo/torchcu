
import torch
import torch.nn.functional as F

def time_stretch_pairwise_distance(input_tensor: torch.Tensor, time_stretch_factor: float, pad_mode: str = 'constant', value: float = 0.0) -> torch.Tensor:
    """
    Performs time stretching on the input tensor, calculates pairwise distances, and returns the result.

    Args:
        input_tensor: The input tensor of shape (B, T, F).
        time_stretch_factor: The factor to stretch the time dimension by.
        pad_mode: The padding mode to use. Can be 'constant', 'reflect', 'replicate', or 'circular'. Defaults to 'constant'.
        value: The value to use for constant padding. Defaults to 0.0.

    Returns:
        A tensor of shape (B, T, T) containing the pairwise distances.
    """
    # Time stretch the input tensor
    stretched_tensor = F.interpolate(input_tensor, size=(int(input_tensor.shape[1] * time_stretch_factor),), mode='linear', align_corners=False)
    
    # Pad the stretched tensor
    stretched_tensor = F.pad(stretched_tensor, (0, 0, 0, int(input_tensor.shape[1] - stretched_tensor.shape[1])), mode=pad_mode, value=value)

    # Calculate pairwise distances
    distances = torch.cdist(stretched_tensor, stretched_tensor, p=2)

    return distances


function_signature = {
    "name": "time_stretch_pairwise_distance",
    "inputs": [
        ((4, 16, 10), torch.float32),
        (torch.float32),
        (str,),
        (torch.float32),
    ],
    "outputs": [
        ((4, 16, 16), torch.float32),
    ]
}
