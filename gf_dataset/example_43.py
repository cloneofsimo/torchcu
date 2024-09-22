
import torch

def torch_softmin_function(input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the softmin of the input tensor along the specified dimension.
    """
    return torch.softmax(-input_tensor, dim=dim)

function_signature = {
    "name": "torch_softmin_function",
    "inputs": [
        ((4, 4), torch.float32),
        (int,)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
