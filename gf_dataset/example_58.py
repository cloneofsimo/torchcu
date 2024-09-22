
import torch
import torch.nn.functional as F

def torch_tanh_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies the tanh function to the input tensor.
    """
    return torch.tanh(input_tensor)

function_signature = {
    "name": "torch_tanh_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
