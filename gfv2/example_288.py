
import torch

def sort_tensor_by_first_dim(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Sorts the input tensor along the first dimension.
    """
    sorted_tensor, indices = torch.sort(input_tensor, dim=0)
    return sorted_tensor

function_signature = {
    "name": "sort_tensor_by_first_dim",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
