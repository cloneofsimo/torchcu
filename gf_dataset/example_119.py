
import torch

def torch_var_backward_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the variance of a tensor and return its gradient.
    """
    mean = torch.mean(input_tensor)
    variance = torch.mean((input_tensor - mean)**2)
    variance.backward()
    return input_tensor.grad

function_signature = {
    "name": "torch_var_backward_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
