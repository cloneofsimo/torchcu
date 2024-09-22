
import torch

def torch_sigmoid_backward_function(input_tensor: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """
    Computes the backward pass for sigmoid activation.
    """
    input_sigmoid = torch.sigmoid(input_tensor)
    grad_input = grad_output * input_sigmoid * (1 - input_sigmoid)
    return grad_input

function_signature = {
    "name": "torch_sigmoid_backward_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
