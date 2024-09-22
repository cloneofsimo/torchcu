
import torch

def torch_tensor_operations(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on a tensor:
      1. Floor the tensor
      2. Check if all elements are greater than 0
      3. Calculate the Frobenius norm
      4. Convert to int8
    """
    floored_tensor = torch.floor(input_tensor)
    all_positive = torch.all(floored_tensor > 0)
    norm = torch.linalg.norm(floored_tensor, ord='fro')
    int8_tensor = floored_tensor.to(torch.int8)
    return int8_tensor

function_signature = {
    "name": "torch_tensor_operations",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.int8),
    ]
}
