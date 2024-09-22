
import torch
import torch.nn.functional as F

def torch_broadcast_add_function(input_tensor: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
    """
    Broadcasts a scalar to the shape of the input tensor and performs element-wise addition.
    """
    return torch.add(input_tensor, scalar.expand_as(input_tensor))

function_signature = {
    "name": "torch_broadcast_add_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((1,), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
