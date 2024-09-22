
import torch

def my_function(input_tensor: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Checks if elements of input_tensor are present in values. 
    Returns a tensor with 1 if the element is in values, 0 otherwise.
    """
    return (input_tensor.unsqueeze(1) == values.unsqueeze(0)).any(dim=1).float()

function_signature = {
    "name": "my_function",
    "inputs": [
        ((4,), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4,), torch.float32)
    ]
}

