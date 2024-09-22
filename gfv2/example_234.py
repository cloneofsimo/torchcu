
import torch

def relu6_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies the ReLU6 activation function element-wise.
    """
    return torch.clamp(input_tensor, min=0, max=6)

function_signature = {
    "name": "relu6_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
