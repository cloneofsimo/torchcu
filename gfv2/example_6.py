
import torch
import torch.nn.functional as F

def upsample_nearest_function(input_tensor: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    Performs nearest neighbor upsampling on the input tensor.
    """
    return F.interpolate(input_tensor, scale_factor=scale_factor, mode='nearest')

function_signature = {
    "name": "upsample_nearest_function",
    "inputs": [
        ((4, 3, 8, 8), torch.float32),
        (1, torch.int32)
    ],
    "outputs": [
        ((4, 3, 16, 16), torch.float32)
    ]
}
