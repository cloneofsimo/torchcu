
import torch

def sigmoid_inplace_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies the sigmoid function inplace to the input tensor.
    """
    input_tensor.sigmoid_()
    return input_tensor

function_signature = {
    "name": "sigmoid_inplace_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
