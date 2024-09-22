
import torch

def my_function(input_tensor: torch.Tensor, scalar: float) -> torch.Tensor:
    """
    This function performs element-wise multiplication with a scalar and then applies a ReLU activation.
    """
    return torch.relu(input_tensor * scalar)

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1, 2, 3), torch.float32),
        (torch.float32,)
    ],
    "outputs": [
        ((1, 2, 3), torch.float32),
    ]
}
