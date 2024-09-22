
import torch

def complex_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations:
        1. Matrix multiplication with 'weight'.
        2. Adds 'bias' to the result.
        3. Applies ReLU activation.
        4. Performs element-wise multiplication with 'input_tensor' (addcmul).
    """
    output = torch.matmul(input_tensor, weight.t())
    output += bias
    output = torch.relu(output)
    output = torch.addcmul(output, 1.0, input_tensor, output)
    return output

function_signature = {
    "name": "complex_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
