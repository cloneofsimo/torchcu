
import torch

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, l2_reg: float) -> torch.Tensor:
    """
    Performs a linear transformation with l2 regularization, applies ReLU activation, and returns the output.
    """
    output = torch.matmul(input_tensor, weight.t()) + bias
    output = torch.relu(output)
    
    # L2 Regularization
    l2_loss = 0.5 * l2_reg * torch.sum(weight ** 2)
    
    # Return output
    return output 

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1,), torch.float32),
        ((1,), torch.float32),
        ((1,), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
