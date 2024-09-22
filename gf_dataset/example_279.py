
import torch

def torch_masked_select_function(input_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Performs masked selection on the input tensor using the provided mask.
    """
    return torch.masked_select(input_tensor, mask)

function_signature = {
    "name": "torch_masked_select_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.bool)  # Mask is boolean
    ],
    "outputs": [
        ((16,), torch.float32)  # Output shape depends on mask
    ]
}
