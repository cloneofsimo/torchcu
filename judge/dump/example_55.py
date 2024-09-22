
import torch

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a simple linear transformation and applies tanh activation.
    """
    output = torch.matmul(input_tensor, weight.t())
    output = torch.tanh(output)
    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
