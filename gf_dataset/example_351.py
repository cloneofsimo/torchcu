
import torch

def torch_sigmoid_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies the sigmoid function to an input tensor.
    """
    return torch.sigmoid(input_tensor)

function_signature = {
    "name": "torch_sigmoid_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
