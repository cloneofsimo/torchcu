
import torch

def my_complex_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Performs a complex operation with layer scaling.
    
    1. Applies a linear transformation (matrix multiplication).
    2. Adds a bias term.
    3. Applies a ReLU activation function.
    4. Scales the output by a given scalar value.
    
    The function also returns the pre-activation output (before ReLU) for backward compatibility. 
    """
    
    pre_activation = torch.matmul(input_tensor, weight.t()) + bias
    output = torch.relu(pre_activation) * scale
    return output, pre_activation

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((1, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((1, 4), torch.float32),
        ((1, 4), torch.float32),
    ]
}
