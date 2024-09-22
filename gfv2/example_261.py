
import torch

def elementwise_min_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise minimum operation on two input tensors.
    """
    return torch.min(input_tensor1, input_tensor2)

function_signature = {
    "name": "elementwise_min_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
