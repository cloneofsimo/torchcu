
import torch

def my_complex_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations including matrix multiplication, bias addition,
    activation, and element-wise summation, with optional bfloat16 conversion.
    """
    output = torch.matmul(input_tensor, weight.t()) + bias
    output = torch.relu(output)  # ReLU activation
    output.add_(torch.rand_like(output, dtype=torch.bfloat16))  # In-place addition with uniform random values
    output = torch.sum(output, dim=1, keepdim=True)  # Sum along the second dimension
    output = output.to(torch.bfloat16)  # Return output in bfloat16
    return output

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((1, 4), torch.float32),  # Input tensor
        ((4, 4), torch.float32),  # Weight tensor
        ((1, 4), torch.float32),  # Bias tensor
    ],
    "outputs": [
        ((1, 1), torch.bfloat16),  # Output tensor
    ]
}
