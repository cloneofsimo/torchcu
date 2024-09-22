
import torch

def torch_eq_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor) -> torch.Tensor:
    """
    Checks if two tensors are equal.
    """
    return torch.eq(input_tensor1, input_tensor2)

function_signature = {
    "name": "torch_eq_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.bool),
    ]
}
