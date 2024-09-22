
import torch

def elementwise_diff_func(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the element-wise difference between two tensors.
    """
    return torch.abs(input_tensor - other_tensor)

function_signature = {
    "name": "elementwise_diff_func",
    "inputs": [
        ((10,), torch.float32),
        ((10,), torch.float32)
    ],
    "outputs": [
        ((10,), torch.float32)
    ]
}
