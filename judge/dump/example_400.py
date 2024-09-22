
import torch

def prelu_backward_example(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Applies PReLU activation and returns the gradient of the output with respect to the input.
    """
    output = torch.where(input_tensor > 0, input_tensor, input_tensor * weight)
    
    # Gradient calculation (for backward pass)
    grad_output = torch.ones_like(output)
    grad_input = torch.where(input_tensor > 0, grad_output, grad_output * weight)
    
    return grad_input

function_signature = {
    "name": "prelu_backward_example",
    "inputs": [
        ((4, 4), torch.float32),
        ((1,), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}

