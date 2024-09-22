
import torch

def torch_pow_inplace_function(input_tensor: torch.Tensor, exponent: float) -> torch.Tensor:
    """
    Perform an inplace power operation on the input tensor.
    """
    input_tensor.pow_(exponent)
    return input_tensor

function_signature = {
    "name": "torch_pow_inplace_function",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
