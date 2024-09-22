
import torch

def torch_generate_int8_data(shape: torch.Size, low: int, high: int) -> torch.Tensor:
    """
    Generates a tensor filled with random integers in the specified range.

    Args:
        shape (torch.Size): The desired shape of the tensor.
        low (int): The lower bound of the range (inclusive).
        high (int): The upper bound of the range (exclusive).

    Returns:
        torch.Tensor: A tensor of the specified shape filled with random integers.
    """
    return torch.randint(low, high, shape, dtype=torch.int8)

function_signature = {
    "name": "torch_generate_int8_data",
    "inputs": [
        ((4, 4), torch.int64),
        (0, torch.int64),
        (10, torch.int64)
    ],
    "outputs": [
        ((4, 4), torch.int8),
    ]
}
