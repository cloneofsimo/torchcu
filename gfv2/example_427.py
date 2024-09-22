
import torch

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise power operation and a linear transformation, then returns the result.
    """
    output = torch.pow(input_tensor, 2.0)  # Element-wise power
    output = torch.matmul(output, weight.t())
    return output.to(torch.bfloat16) 

function_signature = {
    "name": "my_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.bfloat16),
    ]
}
