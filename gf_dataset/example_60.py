
import torch

def torch_identity_int8_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns the input tensor as int8.
    """
    return input_tensor.to(torch.int8)

function_signature = {
    "name": "torch_identity_int8_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.int8)
    ]
}
