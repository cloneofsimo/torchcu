
import torch

def inplace_square_function(input_tensor: torch.Tensor) -> None:
    """
    Squares the elements of the input tensor in-place.
    """
    input_tensor.mul_(input_tensor)

function_signature = {
    "name": "inplace_square_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        
    ]
}
