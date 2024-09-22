
import torch
import torch.nn.functional as F
from torch.nn.functional import pad

def torch_distance_transform_function(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Applies distance transform to the input tensor using a specified kernel size.
    """
    padded_input = pad(input_tensor, (kernel_size // 2, kernel_size // 2), mode="constant", value=0)
    distances = F.conv2d(padded_input.unsqueeze(1), torch.ones(1, 1, kernel_size, kernel_size), padding=0)
    return distances.squeeze(1)

function_signature = {
    "name": "torch_distance_transform_function",
    "inputs": [
        ((10, 10), torch.float32),
        (3, torch.int32)  # Kernel size is a scalar integer
    ],
    "outputs": [
        ((10, 10), torch.float32),
    ]
}
