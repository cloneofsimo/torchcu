
import torch

def torch_relu_inplace_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies the ReLU function to the input tensor inplace.
    """
    torch.relu_(input_tensor)
    return input_tensor

function_signature = {
    "name": "torch_relu_inplace_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
