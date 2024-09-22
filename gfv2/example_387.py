
import torch

def complex_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a complex sequence of operations:
        1.  Matrix multiplication with weight.t()
        2.  Element-wise clipping to the range [-1, 1]
        3.  Cholesky decomposition
        4.  Conversion to int8
        5.  Conversion back to bfloat16
    """
    output = torch.matmul(input_tensor, weight.t())
    output = torch.clip(output, -1, 1)  # Clamp values
    output = torch.cholesky(output)
    output = output.to(torch.int8)  # Convert to int8
    output = output.to(torch.bfloat16)  # Convert back to bfloat16
    return output

function_signature = {
    "name": "complex_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.bfloat16),
    ]
}
