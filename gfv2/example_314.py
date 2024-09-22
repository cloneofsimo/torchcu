
import torch

def abs_inplace_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the absolute value of the input tensor and returns the result in-place.
    """
    torch.abs(input_tensor, out=input_tensor)
    return input_tensor

function_signature = {
    "name": "abs_inplace_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
