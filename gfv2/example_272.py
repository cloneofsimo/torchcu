
import torch

def true_divide_and_relu(input_tensor: torch.Tensor, divisor: float) -> torch.Tensor:
    """
    Performs a true division by a scalar and applies ReLU activation inplace.
    """
    input_tensor.true_divide_(divisor)
    input_tensor.relu_()
    return input_tensor

function_signature = {
    "name": "true_divide_and_relu",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
