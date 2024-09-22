
import torch

def complex_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor:
    1. Matrix multiplication with weight
    2. Addition of bias
    3. Exponential activation
    4. Element-wise multiplication by a scalar (2.0)
    5. Conversion to int8
    """
    output = torch.matmul(input_tensor.to(torch.bfloat16), weight.to(torch.bfloat16).t())
    output = output.to(torch.float32) + bias 
    output = torch.exp(output) * 2.0
    return output.to(torch.int8)

function_signature = {
    "name": "complex_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.int8),
    ]
}
