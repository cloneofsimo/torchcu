
import torch

def torch_sum_inplace(input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Calculate the sum of the input tensor along the specified dimension, in-place.
    """
    input_tensor.sum(dim=dim, out=input_tensor)
    return input_tensor

function_signature = {
    "name": "torch_sum_inplace",
    "inputs": [
        ((4, 4, 4), torch.float32),
        (int, None),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
