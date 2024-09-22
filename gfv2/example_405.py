
import torch

def my_complex_function(input_tensor: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor, including:
        - Logspace calculation
        - Einsum contraction with weights
        - Addition of biases
        - In-place ReLU activation
    """
    # Logspace calculation
    logspace_tensor = torch.logspace(0, 1, input_tensor.shape[-1], base=2, dtype=torch.float32)
    input_tensor = input_tensor * logspace_tensor

    # Einsum contraction with weights
    output = torch.einsum('ijk,kl->ijl', input_tensor, weights)

    # Addition of biases
    output += biases

    # In-place ReLU activation
    output.relu_()
    
    return output

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((4, 5), torch.float32),
        ((5), torch.float32),
    ],
    "outputs": [
        ((2, 3, 5), torch.float32),
    ]
}
