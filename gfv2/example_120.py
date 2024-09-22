
import torch

def mean_int8_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean of an int8 tensor and returns the result as a float32 tensor.
    """
    input_int8 = input_tensor.to(torch.int8)
    mean = input_int8.mean().to(torch.float32)
    return mean

function_signature = {
    "name": "mean_int8_function",
    "inputs": [
        ((4, 4), torch.int8),
    ],
    "outputs": [
        ((), torch.float32),
    ]
}
